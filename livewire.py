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

The module is deliberately Pyodide-agnostic: it runs unchanged under
CPython (see tests/) and inside the browser via a Web Worker. The only
Pyodide accommodation is `_as_bytes_like`, which unwraps JsProxy buffers.
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

# Engine state (one image at a time)
_w = 0
_h = 0
_graph = None
_pred = None
_seed = -1


def _as_bytes_like(buf):
    """Unwrap a Pyodide JsProxy (TypedArray) into a memoryview if needed."""
    if hasattr(buf, "to_py"):
        return buf.to_py()
    return buf


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
