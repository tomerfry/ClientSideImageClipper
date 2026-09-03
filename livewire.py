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

# Engine state (one image at a time)
_w = 0
_h = 0
_graph = None
_pred = None
_seed = -1
_auto_grad = None         # colour-gradient magnitude map (contrast units)
_auto_graph = None        # barrier-cost grid graph, built lazily on first use
_auto_pos = None          # (seed key, distance map) for the positive seeds
_auto_neg = None          # same for the negative seeds, or None
_auto_seed_idx = None     # flat indices of the snapped positive seed pixels


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
    global _auto_grad, _auto_graph, _auto_pos, _auto_neg, _auto_seed_idx
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
    _auto_graph = None
    _auto_pos = None
    _auto_neg = None
    _auto_seed_idx = None


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
# Negative seeds run a second flood; a pixel stays foreground only if it
# is geodesically closer to a positive seed than to any negative one
# (the GeoS rule), which lets a click push back a leaked region.


def _contrast_map(rgb):
    """Colour-gradient magnitude in grey-level units: a step of h levels
    (in all three channels) costs ~h to cross, whatever its blur."""
    acc = np.zeros(rgb.shape[:2], dtype=np.float32)
    for c in range(3):
        s = ndimage.gaussian_filter(rgb[..., c], _AUTO_SIGMA)
        acc += ndimage.sobel(s, axis=1) ** 2 + ndimage.sobel(s, axis=0) ** 2
    # sobel of a unit step sums to 8 across the edge; 3 channels -> sqrt(3)
    return (np.sqrt(acc) / (8.0 * np.sqrt(3.0))).astype(np.float32)


def _ensure_auto_graph():
    """Build the barrier graph on first use (it costs as much as the
    livewire graph, so it's deferred until the user actually shift+clicks)."""
    global _auto_graph
    if _auto_graph is not None:
        return
    if _auto_grad is None:
        raise RuntimeError("set_image must be called first")
    g = _auto_grad
    # soft noise floor: g^2/(g+f) ~ g for real edges, ~g^2/f for texture,
    # so faint noise barely accumulates while wide soft edges keep most of
    # their contrast (a hard floor would let blurry outlines leak)
    floor = float(np.clip(_AUTO_FLOOR_K * np.median(g), _AUTO_FLOOR_LO, _AUTO_FLOOR_HI))
    barrier = g * g / (g + floor)
    _auto_graph = _build_graph((barrier + _AUTO_EPS).astype(np.float64))


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


def _flood(sources):
    """Minimum crossing cost from any source pixel to every pixel
    (inf beyond _AUTO_LIMIT, which lets dijkstra stop early)."""
    d = dijkstra(
        _auto_graph, directed=True, indices=sources,
        min_only=True, limit=_AUTO_LIMIT,
    )
    return d.astype(np.float32).reshape(_h, _w)


def auto_select(pos, neg):
    """Register the seed clicks: `pos`/`neg` are flat [x0, y0, x1, y1, ...]
    lists of work-pixel coordinates (negative seeds may be empty). Runs
    the flood(s) only for the seed set that changed, then returns the
    suggested `reach` (see `_suggest_reach`). Call `auto_mask` next."""
    global _auto_pos, _auto_neg, _auto_seed_idx
    _ensure_auto_graph()
    pos = _as_list(pos)
    neg = _as_list(neg)
    if len(pos) < 2:
        raise ValueError("auto_select needs at least one positive seed")
    pk, nk = tuple(pos), tuple(neg)
    if _auto_pos is None or _auto_pos[0] != pk:
        seed_idx, sources = _seed_pixels(pos)
        _auto_pos = (pk, _flood(sources))
        _auto_seed_idx = seed_idx
    if len(neg) < 2:
        _auto_neg = None
    elif _auto_neg is None or _auto_neg[0] != nk:
        _, sources = _seed_pixels(neg)
        _auto_neg = (nk, _flood(sources))
    return _suggest_reach()


def _suggest_reach():
    """Pick a reach automatically from the region-growth curve A(t) =
    #pixels with d <= t. While t sweeps the object's interior the area
    jumps; once the object is full it sits on a plateau until t exceeds
    the outline's contrast and the flood leaks into the background. We
    return the middle of the first wide plateau (a plateau is where the
    boundary advances by well under a pixel per unit of reach), so a
    soft edge is cut through its middle. Falls back to 30."""
    d = _auto_pos[1]
    ok = np.isfinite(d)
    if _auto_neg is not None:
        ok &= d < _auto_neg[1]
    vals = d[ok]
    total = d.size
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
    d = _auto_pos[1]
    fg = d <= float(reach)
    if _auto_neg is not None:
        fg &= d < _auto_neg[1]
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
