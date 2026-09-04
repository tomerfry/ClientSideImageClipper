"""Livewire ("intelligent scissors") path engine for the magic clipper.

Implements the classic Mortensen & Barrett magnetic-lasso algorithm:

  1. Build a per-pixel *local cost* map from the image — low cost on object
     outlines (strong gradient magnitude, Laplacian zero-crossings), high
     cost on flat regions.
  2. Treat the image as an 8-connected graph whose edge weights are the
     local cost of the target pixel (scaled by step length).
  3. When the user drops an anchor, run Dijkstra from that seed over the
     whole graph (scipy's C implementation) and keep the predecessor tree.
  4. While the cursor moves, extracting the optimal seed->cursor path is
     just a predecessor walk — effectively free, so the path "clings" to
     outlines in real time.

Auto-select (shift+click) reuses the same grid with *inverted* semantics:
stepping onto a pixel costs its colour contrast, so Dijkstra's distance
from the click is "how much edge you must cross to get here" and the
object is simply everything below a contrast budget (see `auto_select`).

The module is deliberately Pyodide-agnostic: it runs unchanged under
CPython (see tests/) and inside the browser via a Web Worker. The only
Pyodide accommodation is `_as_bytes_like` / `_as_list`, which unwrap
JsProxy buffers and arrays.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# 8-connected neighbourhood offsets (dy, dx)
_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_SQRT2 = 1.4142135623730951
_NO_PRED = -9999  # scipy.sparse.csgraph sentinel for "no predecessor"

# Cost-map weights (Mortensen & Barrett use 0.43/0.43/0.14; we fold the
# directional term into a constant base cost that doubles as a length
# penalty, keeping every edge weight strictly positive).
_W_GRAD = 0.55
_W_ZERO = 0.35
_W_BASE = 0.10

# Auto-select tuning
_AUTO_SIGMA = 1.2         # pre-blur before the colour gradient (texture vs. precision)
_AUTO_EPS = 0.01          # per-pixel length cost: keeps weights > 0, bounds floods
_AUTO_SNAP = 3            # a click snaps to the flattest pixel within this radius
_AUTO_FLOOR_K = 2.0       # soft noise floor = clip(K * median gradient, LO, HI)
_AUTO_FLOOR_LO = 2.5
_AUTO_FLOOR_HI = 8.0
_AUTO_REACH_MAX = 100     # the UI slider's range (contrast units, 0..255 scale)
_AUTO_LIMIT = _AUTO_REACH_MAX + 5.0  # dijkstra gives up beyond this distance
# "smart" mode: appearance model on top of the edges
_AUTO_ITERS = 1           # model rounds after the edge-only flood (more = GrabCut-style runaway on similar colours)
_AUTO_BINS = (8, 24, 24, 4)   # luma, chroma-b, chroma-r, texture level
_AUTO_CHROMA_SPAN = 150.0 # chroma axes cover [-span, span] (finer bins tell pastels apart)
_AUTO_TEX_EDGES = (3.0, 8.0, 20.0)  # texture-level bin edges (mean local gradient, contrast units)
_AUTO_SPACE = "ycc"                 # colour axes: "ycc" (luma-difference chroma) or "chromaticity" (shading-invariant)
_AUTO_CHROMATICITY_RANGE = (0.08, 0.6)  # r/(r+g+b), g/(r+g+b) range the chromaticity bins cover
_AUTO_HIST_SIGMA = (1.2, 0.8, 0.8)  # histogram smoothing per (luma, chroma, chroma) axis, in bins (luma blurred: shading)
_AUTO_SHARPEN = 2.0       # likelihood-ratio exponent: pushes P toward 0/1
_AUTO_P_SIGMA = 1.5       # spatial smoothing of the probability map (px)
_AUTO_TOLL = 0.6          # per-pixel cost of walking through background-like pixels
_AUTO_P_SURE = 0.70       # P above this: raw edges are ignored (object interior)
_AUTO_P_DOUBT = 0.50      # P below this: raw edges count in full
_AUTO_RHO_WINDOW = 5      # edge discount judged on the min P in this window (px)

# Engine state (one image at a time)
_w = 0
_h = 0
_graph = None
_pred = None
_seed = -1
_auto_grad = None         # colour-gradient magnitude map (contrast units)
_auto_bins = None         # per-pixel appearance bin (colour + texture), int32
_auto_barrier = None      # soft-floored edge cost map (smart mode reuses it)
_auto_grid = None         # (indptr, indices, step) of the 8-connected grid, built once
_auto_graph = None        # edge-only barrier graph, built lazily on first use
_auto_key = None          # (pos, neg, mode) the cached floods belong to
_auto_pos = None          # distance map from the positive seeds
_auto_neg = None          # distance map from the negative seeds, or None
_auto_seed_idx = None     # flat indices of the snapped positive seed pixels
_auto_prob = None         # last foreground-probability map (smart mode; for tooling)


def _as_bytes_like(buf):
    """Unwrap a Pyodide JsProxy (TypedArray) into a memoryview if needed."""
    if hasattr(buf, "to_py"):
        return buf.to_py()
    return buf


def _as_list(seq):
    """Unwrap a Pyodide JsProxy (Array) into a plain list if needed."""
    if hasattr(seq, "to_py"):
        seq = seq.to_py()
    return [float(v) for v in seq]


def _zero_crossing_cost(lap):
    """0.0 on Laplacian zero-crossing pixels (likely true edge centres),
    1.0 elsewhere. A pixel is a crossing if any 8-neighbour has opposite
    Laplacian sign and this pixel is the smaller-magnitude side."""
    zc = np.ones(lap.shape, dtype=np.float32)
    absl = np.abs(lap)
    for dy, dx in _OFFSETS:
        n_lap = np.roll(np.roll(lap, dy, axis=0), dx, axis=1)
        n_abs = np.roll(np.roll(absl, dy, axis=0), dx, axis=1)
        crossing = (lap * n_lap) < 0
        zc[crossing & (absl <= n_abs)] = 0.0
    # np.roll wraps around the borders; don't let phantom crossings there
    zc[0, :] = zc[-1, :] = 1.0
    zc[:, 0] = zc[:, -1] = 1.0
    return zc


def _build_graph(cost):
    """CSR adjacency of the 8-connected pixel grid; the weight of the edge
    into a pixel is that pixel's local cost times the step length."""
    h, w = cost.shape
    n = h * w
    idx = np.arange(n, dtype=np.int32).reshape(h, w)
    rows, cols, data = [], [], []
    for dy, dx in _OFFSETS:
        step = _SQRT2 if (dy != 0 and dx != 0) else 1.0
        r0, r1 = max(0, -dy), h - max(0, dy)
        c0, c1 = max(0, -dx), w - max(0, dx)
        src = idx[r0:r1, c0:c1]
        dst = idx[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
        wgt = step * cost[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
        rows.append(src.ravel())
        cols.append(dst.ravel())
        data.append(wgt.ravel())
    return csr_matrix(
        (np.concatenate(data),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )


def set_image(rgba, width, height):
    """Ingest an RGBA byte buffer (width*height*4) and precompute the
    cost graph. Must be called before set_seed/get_path."""
    global _w, _h, _graph, _pred, _seed
    global _auto_grad, _auto_bins, _auto_barrier, _auto_grid, _auto_graph
    global _auto_key, _auto_pos, _auto_neg, _auto_seed_idx, _auto_prob
    width = int(width)
    height = int(height)
    buf = np.frombuffer(_as_bytes_like(rgba), dtype=np.uint8)
    px = buf.reshape(height, width, 4).astype(np.float32)
    gray = (0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]) / 255.0

    smooth = ndimage.gaussian_filter(gray, 1.0)
    gx = ndimage.sobel(smooth, axis=1)
    gy = ndimage.sobel(smooth, axis=0)
    gmag = np.hypot(gx, gy)
    peak = float(gmag.max())
    if peak > 0.0:
        gmag /= peak
    inv_grad = 1.0 - gmag  # strong edge -> cheap to walk along

    zc = _zero_crossing_cost(ndimage.gaussian_laplace(gray, 1.0))

    cost = (_W_GRAD * inv_grad + _W_ZERO * zc + _W_BASE).astype(np.float32)

    _w, _h = width, height
    _graph = _build_graph(cost)
    _pred = None
    _seed = -1
    _auto_grad = _contrast_map(px[..., :3])
    _auto_bins = _appearance_bins(px[..., :3], _auto_grad)
    _auto_barrier = None
    _auto_grid = None
    _auto_graph = None
    _auto_key = None
    _auto_pos = None
    _auto_neg = None
    _auto_seed_idx = None
    _auto_prob = None


def set_seed(x, y):
    """Anchor dropped at (x, y): run Dijkstra from it and keep the
    predecessor tree so any subsequent path query is a cheap walk."""
    global _pred, _seed
    if _graph is None:
        raise RuntimeError("set_image must be called first")
    x = min(max(int(x), 0), _w - 1)
    y = min(max(int(y), 0), _h - 1)
    seed = y * _w + x
    _, pred = dijkstra(
        _graph, directed=True, indices=seed, return_predecessors=True
    )
    _pred = pred
    _seed = seed


def trim_cutout(rgba, width, height, tolerance):
    """Auto-trim a finished cutout (RGBA with transparency outside the
    lasso): make background-coloured pixels that are *reachable from the
    outer edge* transparent, drop tiny disconnected specks, and crop to
    the surviving content.

    The background colour is estimated as the median colour of the
    cutout's opaque rim — exactly where leftover background lives. Only
    regions connected to the outside are removed, so background-coloured
    pixels *inside* the subject survive.

    Returns (rgba_bytes, x, y, w, h): crop offset within the input and
    the crop size. (b"", 0, 0, 0, 0) if nothing survives.
    """
    width = int(width)
    height = int(height)
    a = np.frombuffer(_as_bytes_like(rgba), dtype=np.uint8) \
          .reshape(height, width, 4).copy()
    alpha = a[..., 3]
    opaque = alpha > 8
    if not opaque.any():
        return (b"", 0, 0, 0, 0)

    eight = np.ones((3, 3), dtype=bool)
    rim = opaque & ~ndimage.binary_erosion(opaque, structure=eight, border_value=0)
    rgb = a[..., :3].astype(np.float32)
    bg = np.median(rgb[rim], axis=0)

    bg_like = np.sqrt(((rgb - bg) ** 2).sum(axis=-1)) <= float(tolerance)
    outside = ~opaque
    lab, _ = ndimage.label(bg_like | outside, structure=eight)
    frame = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    kill = np.unique(np.concatenate([lab[outside].ravel(), frame]))
    kill = kill[kill != 0]
    removed = np.isin(lab, kill) & opaque
    a[..., 3][removed] = 0
    # feather the freshly cut edge (the original polygon edge keeps its AA)
    soft = ndimage.gaussian_filter((~removed).astype(np.float32), 0.6)
    a[..., 3] = np.minimum(a[..., 3], (soft * 255.0).astype(np.uint8))

    # drop disconnected specks far smaller than the main subject
    solid = a[..., 3] > 0
    lab2, n2 = ndimage.label(solid, structure=eight)
    if n2 > 1:
        sizes = np.bincount(lab2.ravel(), minlength=n2 + 1)
        sizes[0] = 0
        small = np.flatnonzero(sizes < max(16, 0.08 * sizes.max()))
        a[..., 3][np.isin(lab2, small)] = 0
        solid = a[..., 3] > 0

    return _crop_to_content(a)


def _crop_to_content(a):
    """Crop an RGBA array to its alpha>0 bounding box (+1px pad) and
    return the (bytes, x, y, w, h) tuple the worker protocol expects."""
    height, width = a.shape[:2]
    solid = a[..., 3] > 0
    if not solid.any():
        return (b"", 0, 0, 0, 0)
    ys, xs = np.nonzero(solid)
    x0 = max(0, int(xs.min()) - 1)
    x1 = min(width, int(xs.max()) + 2)
    y0 = max(0, int(ys.min()) - 1)
    y1 = min(height, int(ys.max()) + 2)
    crop = np.ascontiguousarray(a[y0:y1, x0:x1])
    return (crop.tobytes(), x0, y0, x1 - x0, y1 - y0)


def smooth_edges(rgba, width, height, amount):
    """Smooth a cutout's outline: round jagged, stair-stepped alpha
    boundaries while keeping the edge crisp (~1px anti-aliased ramp).

    Method: extend the object's colours into transparent pixels (nearest
    opaque pixel via distance transform, so the reshaped edge never shows
    a dark fringe), Gaussian-blur the alpha channel, then re-steepen the
    ramp around 0.5 — a blur-and-sharpen contour smoothing, like
    Photoshop's mask "Smooth".

    `amount` is the UI slider value (1..10). Returns the standard
    (rgba_bytes, x, y, w, h) crop tuple.
    """
    width = int(width)
    height = int(height)
    a = np.frombuffer(_as_bytes_like(rgba), dtype=np.uint8) \
          .reshape(height, width, 4).copy()
    sigma = 0.3 + 0.4 * float(amount)
    alpha = a[..., 3].astype(np.float32) / 255.0
    if amount <= 0 or not (alpha > 0).any():
        return _crop_to_content(a)

    # pad so the reshaped contour can move past the original bounds
    p = int(np.ceil(sigma)) + 2
    a = np.pad(a, ((p, p), (p, p), (0, 0)))
    alpha = np.pad(alpha, p)

    solid = alpha > 0.25
    if solid.any() and (~solid).any():
        ind = ndimage.distance_transform_edt(
            ~solid, return_distances=False, return_indices=True
        )
        a[..., :3] = a[..., :3][ind[0], ind[1]]

    blurred = ndimage.gaussian_filter(alpha, sigma)
    steep = max(1.6, 2.1 * sigma)  # keep the final edge ~1px wide
    out = np.clip((blurred - 0.5) * steep + 0.5, 0.0, 1.0)
    a[..., 3] = (out * 255.0 + 0.5).astype(np.uint8)
    data, x0, y0, w, h = _crop_to_content(a)
    return (data, x0 - p, y0 - p, w, h)


# ── auto-select: click-to-object segmentation ──────────────────────────
#
# A geodesic "barrier flood" from the clicked pixel(s). The image is the
# same 8-connected grid as the livewire, but the cost of stepping onto a
# pixel is its colour-gradient magnitude, so a path's total cost is about
# the amount of contrast it has to cross (the integral of the gradient
# across an edge equals the edge's contrast, however blurry it is).
# Dijkstra from the seed then gives every pixel its minimum crossing
# cost d(p); the object is the sublevel set {p : d(p) <= reach}. Flat and
# mildly textured interiors cost ~nothing, real outlines cost their
# contrast: the region fills the object and stops at its boundary, and
# `reach` (the UI slider) is the contrast budget. Re-thresholding is
# free, so adjusting the selection never re-runs Dijkstra.
#
# "smart" mode (default) adds an appearance model, GrabCut-style but
# without the graph cut: the edge-only region seeds a foreground
# colour+texture histogram, the rest of the image a background one, and
# every pixel gets a foreground probability P. The flood is re-run on a
# cost that charges raw edges only where the pixel does not look like
# the object — so creases, shading and texture *inside* it are free —
# and tolls every step through background-looking pixels, so a leak
# through a gap in the outline runs out of budget. One round only: a
# second round re-samples from the grown region and, like GrabCut on
# similar colours, tends to run away. The flip side of trusting the
# model is that two touching regions of the same appearance merge even
# across a real edge; a negative seed (or "edges" mode) separates them.
#
# Negative seeds run a second flood; a pixel stays foreground only if it
# is geodesically closer to a positive seed than to any negative one
# (the GeoS rule), which lets a click push back a leaked region. They
# also feed the background model.


def _contrast_map(rgb):
    """Colour-gradient magnitude in grey-level units: a step of h levels
    (in all three channels) costs ~h to cross, whatever its blur."""
    acc = np.zeros(rgb.shape[:2], dtype=np.float32)
    for c in range(3):
        s = ndimage.gaussian_filter(rgb[..., c], _AUTO_SIGMA)
        acc += ndimage.sobel(s, axis=1) ** 2 + ndimage.sobel(s, axis=0) ** 2
    # sobel of a unit step sums to 8 across the edge; 3 channels -> sqrt(3)
    return (np.sqrt(acc) / (8.0 * np.sqrt(3.0))).astype(np.float32)


def _appearance_bins(rgb, grad):
    """Quantise every pixel into a (luma, chroma, chroma, texture) bin.
    Texture = local gradient energy averaged over a few pixels, so a
    leafy or woven surface is one appearance even though its colours
    swing pixel to pixel."""
    nl, nb, nr, nt = _AUTO_BINS
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    li = np.clip((y / 256.0 * nl).astype(np.int32), 0, nl - 1)
    if _AUTO_SPACE == "chromaticity":
        # r/(r+g+b), g/(r+g+b): invariant to shading, so a lit and a
        # shadowed part of the same surface share a bin
        ssum = r + g + b + 3.0
        lo, hi = _AUTO_CHROMATICITY_RANGE
        bi = np.clip(((r + 1.0) / ssum - lo) / (hi - lo) * nb, 0, nb - 1).astype(np.int32)
        ri = np.clip(((g + 1.0) / ssum - lo) / (hi - lo) * nr, 0, nr - 1).astype(np.int32)
    else:
        cb = b - y
        cr = r - y
        span = float(_AUTO_CHROMA_SPAN)
        bi = np.clip(((cb + span) / (2 * span) * nb).astype(np.int32), 0, nb - 1)
        ri = np.clip(((cr + span) / (2 * span) * nr).astype(np.int32), 0, nr - 1)
    tex = ndimage.gaussian_filter(grad, 3.0)
    ti = np.digitize(tex, list(_AUTO_TEX_EDGES)[:nt - 1]).astype(np.int32)
    return ((li * nb + bi) * nr + ri) * nt + ti


def _ensure_auto_graph():
    """Build the grid structure and the edge-only barrier graph on first
    use (it costs as much as the livewire graph, so it's deferred until
    the user actually shift+clicks)."""
    global _auto_graph, _auto_grid, _auto_barrier
    if _auto_graph is not None:
        return
    if _auto_grad is None:
        raise RuntimeError("set_image must be called first")
    g = _auto_grad
    # soft noise floor: g^2/(g+f) ~ g for real edges, ~g^2/f for texture,
    # so faint noise barely accumulates while wide soft edges keep most of
    # their contrast (a hard floor would let blurry outlines leak)
    floor = float(np.clip(_AUTO_FLOOR_K * np.median(g), _AUTO_FLOOR_LO, _AUTO_FLOOR_HI))
    _auto_barrier = (g * g / (g + floor)).astype(np.float64)
    ones = _build_graph(np.ones((_h, _w), dtype=np.float64))
    _auto_grid = (ones.indptr, ones.indices, ones.data)   # data = step length
    _auto_graph = _graph_from_cost(_auto_barrier + _AUTO_EPS)


def _graph_from_cost(cost):
    """CSR grid graph with per-pixel entry cost, sharing the structure
    built once per image (only the weights are recomputed)."""
    indptr, indices, step = _auto_grid
    data = step * cost.ravel()[indices]
    n = _w * _h
    return csr_matrix((data, indices, indptr), shape=(n, n))


def auto_ready():
    """True once the barrier graph exists (the UI shows a longer busy
    message for the first auto-select on an image)."""
    return _auto_graph is not None


def _seed_pixels(flat):
    """Snap each clicked (x, y) to the flattest nearby pixel (so a click
    that lands on an outline still starts inside the object) and return
    (snapped flat indices, flat indices of their 3x3 source disks)."""
    seeds, sources = [], []
    for i in range(0, len(flat) - 1, 2):
        x = min(max(int(flat[i]), 0), _w - 1)
        y = min(max(int(flat[i + 1]), 0), _h - 1)
        r = _AUTO_SNAP
        y0, y1 = max(0, y - r), min(_h, y + r + 1)
        x0, x1 = max(0, x - r), min(_w, x + r + 1)
        win = _auto_grad[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        score = win + 0.5 * np.hypot(yy - y, xx - x)  # prefer close *and* flat
        k = int(np.argmin(score))
        sy, sx = y0 + k // win.shape[1], x0 + k % win.shape[1]
        seeds.append(sy * _w + sx)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                py, px_ = sy + dy, sx + dx
                if 0 <= py < _h and 0 <= px_ < _w:
                    sources.append(py * _w + px_)
    return np.array(seeds, dtype=np.int64), np.unique(np.array(sources, dtype=np.int64))


def _flood(graph, sources):
    """Minimum crossing cost from any source pixel to every pixel
    (inf beyond _AUTO_LIMIT, which lets dijkstra stop early)."""
    d = dijkstra(
        graph, directed=True, indices=sources,
        min_only=True, limit=_AUTO_LIMIT,
    )
    return d.astype(np.float32).reshape(_h, _w)


def _fg_probability(d_pos, d_neg, reach):
    """Foreground probability per pixel from colour+texture histograms:
    the current region (d_pos <= reach, softly beyond) is the foreground
    sample, everything else — and anything the negative seeds reached —
    the background sample. P = pF^k / (pF^k + pB^k), the likelihood ratio
    sharpened so 'mostly object-coloured' reads as confidently object
    even while the object still pollutes the background sample."""
    r = max(float(reach), 1.0)
    fin = np.isfinite(d_pos)
    dp = np.where(fin, d_pos, np.inf)
    # foreground sample: the region at `reach`, fading to nothing a
    # quarter-reach beyond it (a longer tail would let a neighbouring
    # object that is barely out of reach pollute the foreground model)
    w_f = np.clip(1.0 - (dp - r) / (0.25 * r), 0.0, 1.0)
    w_f[~fin] = 0.0
    w_b = 1.0 - w_f
    if d_neg is not None:
        neg = np.isfinite(d_neg) & (d_neg <= r)
        w_f[neg] = 0.0
        w_b[neg] = 1.0
    nb = int(np.prod(_AUTO_BINS))
    flat = _auto_bins.ravel()
    h_f = np.bincount(flat, weights=w_f.ravel(), minlength=nb).reshape(_AUTO_BINS)
    h_b = np.bincount(flat, weights=w_b.ravel(), minlength=nb).reshape(_AUTO_BINS)
    sig = tuple(_AUTO_HIST_SIGMA) + (0.0,)
    h_f = ndimage.gaussian_filter(h_f, sig)
    h_b = ndimage.gaussian_filter(h_b, sig)
    h_f /= max(h_f.sum(), 1e-9)
    h_b /= max(h_b.sum(), 1e-9)
    eps = 1e-3 / nb
    pf = (h_f + eps) ** _AUTO_SHARPEN
    pb = (h_b + eps) ** _AUTO_SHARPEN
    p_bins = pf / (pf + pb)
    prob = p_bins.ravel()[flat].reshape(_h, _w).astype(np.float32)
    return ndimage.gaussian_filter(prob, _AUTO_P_SIGMA)


def _smart_cost(prob):
    """Per-pixel entry cost combining the appearance model with the raw
    edges: edges count in full where P says 'not the object', fade out
    as P rises to _AUTO_P_SURE, and background-like pixels carry a
    per-step toll. An uninformative model (P ~ 0.5 everywhere) reduces
    to the edge-only cost."""
    # an edge ramp straddles its boundary: judge it by the *least*
    # object-like pixel nearby, so the whole ramp counts next to
    # background while creases deep inside the object stay free
    near = ndimage.minimum_filter(prob, size=_AUTO_RHO_WINDOW)
    rho = np.clip((_AUTO_P_SURE - near) / (_AUTO_P_SURE - _AUTO_P_DOUBT), 0.0, 1.0)
    bg = np.clip((_AUTO_P_DOUBT - prob) / _AUTO_P_DOUBT, 0.0, 1.0)
    cost = rho * _auto_barrier + _AUTO_TOLL * bg + _AUTO_EPS
    return cost.astype(np.float64)


def auto_select(pos, neg, mode="smart"):
    """Register the seed clicks: `pos`/`neg` are flat [x0, y0, x1, y1, ...]
    lists of work-pixel coordinates (negative seeds may be empty), `mode`
    is "smart" (edges + appearance model) or "edges" (outline only).
    Recomputes the floods when anything changed, then returns the
    suggested `reach` (see `_suggest_reach`). Call `auto_mask` next."""
    global _auto_key, _auto_pos, _auto_neg, _auto_seed_idx, _auto_prob
    _ensure_auto_graph()
    pos = _as_list(pos)
    neg = _as_list(neg)
    mode = str(mode)
    if len(pos) < 2:
        raise ValueError("auto_select needs at least one positive seed")
    key = (tuple(pos), tuple(neg), mode)
    if _auto_key == key:
        return _suggest_reach(_auto_pos, _auto_neg)
    seed_idx, src_pos = _seed_pixels(pos)
    src_neg = _seed_pixels(neg)[1] if len(neg) >= 2 else None
    graph = _auto_graph
    d_pos = _flood(graph, src_pos)
    d_neg = _flood(graph, src_neg) if src_neg is not None else None
    prob = None
    if mode == "smart":
        for _ in range(_AUTO_ITERS):
            reach = _suggest_reach(d_pos, d_neg)
            prob = _fg_probability(d_pos, d_neg, reach)
            graph = _graph_from_cost(_smart_cost(prob))
            d_pos = _flood(graph, src_pos)
            d_neg = _flood(graph, src_neg) if src_neg is not None else None
    _auto_key = key
    _auto_pos = d_pos
    _auto_neg = d_neg
    _auto_seed_idx = seed_idx
    _auto_prob = prob
    return _suggest_reach(d_pos, d_neg)


def _suggest_reach(d_pos, d_neg):
    """Pick a reach automatically from the region-growth curve A(t) =
    #pixels with d <= t. While t sweeps the object's interior the area
    jumps; once the object is full it sits on a plateau until t exceeds
    the outline's contrast and the flood leaks into the background. We
    return the middle of the first wide plateau (a plateau is where the
    boundary advances by well under a pixel per unit of reach), so a
    soft edge is cut through its middle. Falls back to 30."""
    ok = np.isfinite(d_pos)
    if d_neg is not None:
        ok &= d_pos < d_neg
    vals = d_pos[ok]
    total = d_pos.size
    edges = np.arange(0, _AUTO_REACH_MAX + 2, dtype=np.float64)
    counts, _ = np.histogram(vals, bins=edges)
    area = np.cumsum(counts).astype(np.float64)       # area[t] ~ A(t)
    tmax = _AUTO_REACH_MAX
    growth = np.full(tmax + 1, np.inf)
    for t in range(2, tmax - 1):
        growth[t] = (area[t + 2] - area[t - 2]) / 4.0
    calm = growth < (1.0 * np.sqrt(area) + 2.0)        # < ~0.3 px boundary advance per unit
    best = None
    t = 2
    while t < tmax - 1:
        if not calm[t]:
            t += 1
            continue
        a = t
        while t < tmax - 1 and calm[t]:
            t += 1
        b = t - 1
        if b - a >= 6 and area[a] >= 100 and area[b] <= 0.85 * total:
            best = (a, b)
            break
    if best is None:
        return 30
    a, b = best
    return int(round(a + 0.5 * (b - a)))


def _clean_mask(fg, seeds):
    """Keep only the components that contain a positive seed, close
    1-px cracks, and fill holes that are tiny relative to the region."""
    eight = np.ones((3, 3), dtype=bool)
    lab, n = ndimage.label(fg, structure=eight)
    if n == 0:
        return fg
    keep = np.unique(lab.ravel()[seeds])
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros_like(fg)
    if n > keep.size:
        lut = np.zeros(n + 1, dtype=bool)
        lut[keep] = True
        fg = lut[lab]
    fg = ndimage.binary_erosion(
        ndimage.binary_dilation(fg, structure=eight), structure=eight, border_value=1
    )
    filled = ndimage.binary_fill_holes(fg)
    holes = filled & ~fg
    if holes.any():
        hl, hn = ndimage.label(holes)
        sizes = np.bincount(hl.ravel(), minlength=hn + 1)
        small = sizes < max(24, 0.02 * fg.sum())
        small[0] = False
        fg = fg | small[hl]
    return fg


def _trace_contours(mask):
    """Every boundary loop of a boolean mask as a polygon through pixel
    *corners* (crack following with the foreground on the left; saddle
    corners turn right, i.e. the foreground is 8-connected). Holes come
    out as their own loops, so an even-odd fill reproduces the mask.
    Returns (float32 (N, 2) corner coords, [vertex count per loop])."""
    p = np.pad(mask, 1)
    H, W = p.shape
    # horizontal cracks: corner row r+1, between pixel rows r and r+1
    above, below = p[:-1, :], p[1:, :]
    hr, hc = np.nonzero(above != below)
    h_east = above[hr, hc]                       # fg above -> walk east
    # vertical cracks: corner column c+1, between pixel columns c and c+1
    left, right = p[:, :-1], p[:, 1:]
    vr, vc = np.nonzero(left != right)
    v_north = left[vr, vc]                       # fg on the west -> walk north
    sx = np.concatenate([np.where(h_east, hc, hc + 1), vc + 1])
    sy = np.concatenate([hr + 1, np.where(v_north, vr + 1, vr)])
    ex = np.concatenate([np.where(h_east, hc + 1, hc), vc + 1])
    ey = np.concatenate([hr + 1, np.where(v_north, vr, vr + 1)])
    dr = np.concatenate([np.where(h_east, 0, 2), np.where(v_north, 3, 1)])  # E S W N
    n = sx.size
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), []
    out = np.full((H + 1, W + 1, 4), -1, dtype=np.int32)
    out[sy, sx, dr] = np.arange(n, dtype=np.int32)
    nxt = out[ey, ex, (dr + 1) % 4]              # right turn first
    m = nxt < 0
    nxt[m] = out[ey[m], ex[m], dr[m]]            # else straight on
    m = nxt < 0
    nxt[m] = out[ey[m], ex[m], (dr[m] + 3) % 4]  # else left turn
    assert (nxt >= 0).all(), "open boundary (cannot happen for a mask)"
    nxt_l = nxt.tolist()
    seen = bytearray(n)
    loops, lens = [], []
    for s in range(n):
        if seen[s]:
            continue
        j, loop = s, []
        while not seen[j]:
            seen[j] = 1
            loop.append(j)
            j = nxt_l[j]
        idx = np.array(loop, dtype=np.int64)
        d = dr[idx]
        corner = d != np.roll(d, 1)              # direction changed -> polygon vertex
        pts = np.stack([sx[idx][corner], sy[idx][corner]], axis=1) - 1   # unpad
        if len(pts) >= 3:
            loops.append(pts)
            lens.append(int(len(pts)))
    if not loops:
        return np.zeros((0, 2), dtype=np.float32), []
    return np.concatenate(loops).astype(np.float32), lens


def auto_mask(reach):
    """Threshold the current flood at `reach`, tidy the mask, and return
    (corner_coords_f32_bytes, [vertex count per loop], area_px). Corner
    coordinates are in work-pixel units where pixel (x, y) spans
    [x, x+1] x [y, y+1]; loops include holes (even-odd fill)."""
    fg = auto_mask_array(reach)
    if fg is None:
        return (b"", [], 0)
    area = int(fg.sum())
    if area == 0:
        return (b"", [], 0)
    coords, lens = _trace_contours(fg)
    return (coords.tobytes(), lens, area)


def auto_mask_array(reach):
    """The cleaned boolean mask at `reach` (None before any seed)."""
    if _auto_pos is None:
        return None
    fg = _auto_pos <= float(reach)
    if _auto_neg is not None:
        fg &= _auto_pos < _auto_neg
    return _clean_mask(fg, _auto_seed_idx)


def get_path(x, y):
    """Optimal path from the current seed to (x, y), as a flat
    [x0, y0, x1, y1, ...] list of pixel coordinates, seed first."""
    if _pred is None:
        return []
    x = min(max(int(x), 0), _w - 1)
    y = min(max(int(y), 0), _h - 1)
    node = y * _w + x
    pts = []
    remaining = _w * _h  # hard bound; the tree has no cycles
    while remaining > 0:
        pts.append((node % _w, node // _w))
        if node == _seed:
            break
        node = int(_pred[node])
        if node == _NO_PRED:  # unreachable (cannot happen on a grid)
            return []
        remaining -= 1
    pts.reverse()
    flat = []
    for px_, py_ in pts:
        flat.append(int(px_))
        flat.append(int(py_))
    return flat
