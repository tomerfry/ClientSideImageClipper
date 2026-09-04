"""Sanity tests for the livewire engine (plain CPython, no browser).

Run:  python tests/test_livewire.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import livewire


def make_rgba(gray_f32):
    """Grayscale float [0,1] image -> flat RGBA uint8 buffer."""
    g = (np.clip(gray_f32, 0.0, 1.0) * 255).astype(np.uint8)
    h, w = g.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g
    rgba[..., 3] = 255
    return rgba.reshape(-1)


def test_path_clings_to_disk_outline():
    """Seed and target sit on a disk's outline a quarter-turn apart; the
    optimal path must follow the circular edge, not cut across."""
    size, cx, cy, r = 300, 150, 150, 80
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (((xx - cx) ** 2 + (yy - cy) ** 2) < r * r).astype(np.float32)
    gray = 0.15 + 0.65 * disk
    rng = np.random.default_rng(42)
    gray += rng.normal(0, 0.01, gray.shape).astype(np.float32)  # mild noise

    livewire.set_image(make_rgba(gray), size, size)
    livewire.set_seed(cx + r, cy)          # 3 o'clock on the outline
    flat = livewire.get_path(cx, cy + r)   # 6 o'clock on the outline

    assert len(flat) >= 4, "path should have multiple points"
    pts = np.array(flat, dtype=np.float64).reshape(-1, 2)
    assert tuple(pts[0]) == (cx + r, cy), f"path must start at seed, got {pts[0]}"
    assert tuple(pts[-1]) == (cx, cy + r), f"path must end at target, got {pts[-1]}"

    radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    on_edge = np.abs(radii - r) <= 4.0
    frac = on_edge.mean()
    arc_len = np.pi * r / 2  # quarter circumference ~ 126 px
    print(f"  disk: {len(pts)} pts, {frac * 100:.1f}% within 4px of the outline, "
          f"radius spread [{radii.min():.1f}, {radii.max():.1f}] (r={r})")
    assert frac > 0.9, f"path wandered off the outline ({frac * 100:.1f}% on edge)"
    assert len(pts) > 0.8 * arc_len, "path suspiciously short for a quarter arc"


def test_flat_image_gives_straight_path():
    """With no edges to cling to, the cheapest path is a straight line."""
    size = 160
    gray = np.full((size, size), 0.5, dtype=np.float32)
    livewire.set_image(make_rgba(gray), size, size)
    livewire.set_seed(20, 80)
    flat = livewire.get_path(140, 80)
    pts = np.array(flat, dtype=np.float64).reshape(-1, 2)
    assert tuple(pts[0]) == (20, 80) and tuple(pts[-1]) == (140, 80)
    dev = np.abs(pts[:, 1] - 80).max()
    print(f"  flat: {len(pts)} pts, max vertical deviation {dev:.0f}px")
    assert dev <= 1.0, f"path should be straight on a flat image (deviation {dev})"
    assert len(pts) == 121, f"straight path should be 121 pts, got {len(pts)}"


def test_trim_cutout_removes_background_and_specks():
    """A cutout with leftover light background around a dark subject and a
    stray speck: trim must clear the background, keep the subject intact,
    drop the speck, and crop to the subject's bbox."""
    w, h = 200, 150
    img = np.zeros((h, w, 4), np.uint8)             # transparent canvas
    img[20:130, 20:180] = (245, 245, 240, 255)      # opaque leftover bg
    img[50:100, 40:120, :3] = (40, 45, 50)          # the subject
    img[30:33, 160:170, :3] = (60, 60, 60)          # stray speck (30 px)

    data, x0, y0, cw, ch = livewire.trim_cutout(img.reshape(-1), w, h, 40)
    out = np.frombuffer(data, np.uint8).reshape(ch, cw, 4)
    print(f"  trim: crop at ({x0},{y0}) size {cw}x{ch} (subject bbox 40..120 x 50..100)")

    assert abs(x0 - 39) <= 2 and abs(y0 - 49) <= 2, f"crop origin off: ({x0},{y0})"
    assert abs(cw - 82) <= 4 and abs(ch - 52) <= 4, f"crop size off: {cw}x{ch}"
    solid = out[..., 3] > 128
    assert solid.sum() >= 0.9 * (80 * 50), "subject lost pixels"
    mean_rgb = out[..., :3][solid].mean(axis=0)
    assert mean_rgb.max() < 100, f"background survived the trim: {mean_rgb}"

    # background-coloured pixels NOT reachable from outside must survive:
    img2 = img.copy()
    img2[60:90, 60:100, :3] = (245, 245, 240)       # bg-coloured hole inside subject
    data2, _, _, cw2, ch2 = livewire.trim_cutout(img2.reshape(-1), w, h, 40)
    out2 = np.frombuffer(data2, np.uint8).reshape(ch2, cw2, 4)
    bright = (out2[..., :3].astype(int).sum(-1) > 600) & (out2[..., 3] > 128)
    assert bright.sum() >= 0.9 * (30 * 40), "interior bg-coloured region was wrongly removed"
    print(f"  trim: interior bg-coloured region preserved ({bright.sum()} px)")


def test_smooth_edges_rounds_jaggies_without_fringe():
    """A disk with a deliberately ragged 1-2px boundary: smoothing must
    shorten the contour (fewer jaggies), roughly preserve area, and the
    anti-aliased edge pixels must keep the object's colour (no dark
    fringe from transparent-black neighbours)."""
    from scipy import ndimage
    w = h = 160
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(3)
    wobble = rng.integers(-2, 3, size=(h, w))          # ragged boundary
    disk = (np.hypot(xx - 80, yy - 80) + wobble) < 50
    img = np.zeros((h, w, 4), np.uint8)
    img[disk] = (250, 120, 30, 255)                     # orange on transparent

    def contour_len(alpha):
        s = alpha > 128
        return (s[:, 1:] != s[:, :-1]).sum() + (s[1:, :] != s[:-1, :]).sum()

    len_before = contour_len(img[..., 3])
    area_before = (img[..., 3] > 128).sum()

    data, x0, y0, cw, ch = livewire.smooth_edges(img.reshape(-1), w, h, 5)
    out = np.frombuffer(data, np.uint8).reshape(ch, cw, 4)
    len_after = contour_len(out[..., 3])
    area_after = (out[..., 3] > 128).sum()
    print(f"  smooth: contour {len_before} -> {len_after} transitions, "
          f"area {area_before} -> {area_after}")
    assert len_after < 0.9 * len_before, "outline did not get smoother"
    assert abs(area_after - area_before) < 0.1 * area_before, "area drifted too much"

    semi = (out[..., 3] > 30) & (out[..., 3] < 225)
    assert semi.sum() > 50, "expected an anti-aliased edge band"
    edge_rgb = out[..., :3][semi].astype(float).mean(axis=0)
    print(f"  smooth: {semi.sum()} AA edge px, mean colour {edge_rgb.round(0)}")
    assert abs(edge_rgb[0] - 250) < 25 and abs(edge_rgb[2] - 30) < 25, \
        f"edge colour bled: {edge_rgb}"


def bench_browser_resolution():
    """Timing at the app's working resolution (WORK_MAX=768). The browser
    (WASM) is roughly 2-3x slower than native — keep an eye on these."""
    w, h = 768, 576
    rng = np.random.default_rng(7)
    gray = ndimage_blur(rng.random((h, w)).astype(np.float32))

    t0 = time.perf_counter()
    livewire.set_image(make_rgba(gray), w, h)
    t1 = time.perf_counter()
    livewire.set_seed(w // 4, h // 4)
    t2 = time.perf_counter()
    livewire.get_path(3 * w // 4, 3 * h // 4)
    t3 = time.perf_counter()
    print(f"  bench {w}x{h}: set_image {t1 - t0:.2f}s | "
          f"set_seed (dijkstra) {t2 - t1:.2f}s | get_path {(t3 - t2) * 1000:.1f}ms")


# ── auto-select ────────────────────────────────────────────────────

def rgba_from_rgb(rgb):
    """(h, w, 3) uint8 -> flat RGBA buffer."""
    h, w = rgb.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = 255
    return out.reshape(-1)


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def polygon_area(coords, lens):
    """Shoelace area of every loop. With the foreground on the left in
    y-down screen coordinates outer loops run clockwise (negative
    shoelace) and holes counter-clockwise, so -sum equals the mask area."""
    total, o = 0.0, 0
    for n in lens:
        p = coords[o:o + n]
        x, y = p[:, 0], p[:, 1]
        total += 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        o += n
    return -total


def test_auto_select_disk_single_click():
    """A shaded, slightly noisy disk on a noisy background: one click in
    the middle must select the whole disk (and nothing else) at the
    automatically chosen reach, and the traced contour must enclose it."""
    from scipy import ndimage
    size, cx, cy, r = 320, 160, 150, 90
    yy, xx = np.mgrid[0:size, 0:size]
    disk = np.hypot(xx - cx, yy - cy) < r
    rng = np.random.default_rng(1)
    shade = 0.15 * (xx - cx) / r                                  # smooth shading inside
    rgb = np.empty((size, size, 3), np.float32)
    rgb[..., 0] = np.where(disk, 60 + 40 * shade, 200)
    rgb[..., 1] = np.where(disk, 110 + 40 * shade, 205)
    rgb[..., 2] = np.where(disk, 200 + 40 * shade, 210)
    rgb += rng.normal(0, 3.0, rgb.shape)
    rgb = ndimage.gaussian_filter(rgb, (0.8, 0.8, 0))              # soften the outline a bit
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    livewire.set_image(rgba_from_rgb(rgb), size, size)
    t0 = time.perf_counter()
    reach = livewire.auto_select([cx, cy], [])
    t1 = time.perf_counter()
    buf, lens, area = livewire.auto_mask(reach)
    t2 = time.perf_counter()
    mask = livewire.auto_mask_array(reach)
    score = iou(mask, disk)
    print(f"  disk: auto reach {reach}, area {area} (disk {disk.sum()}), IoU {score:.3f}, "
          f"{len(lens)} loop(s) / {sum(lens)} verts | select {t1 - t0:.2f}s mask {(t2 - t1) * 1000:.0f}ms")
    assert 8 <= reach <= 95, f"auto reach out of range: {reach}"
    assert score > 0.95, f"disk not selected cleanly (IoU {score:.3f})"
    assert len(lens) == 1, f"expected one boundary loop, got {len(lens)}"
    coords = np.frombuffer(buf, np.float32).reshape(-1, 2)
    poly_area = polygon_area(coords, lens)
    assert abs(poly_area - area) < 1e-3 * area + 1, f"contour area {poly_area} != mask area {area}"
    radii = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
    assert np.abs(radii - r).max() < 4.0, f"contour strays from the outline: {np.abs(radii - r).max():.1f}px"

    # the reach slider grows/shrinks smoothly: monotone area, no leak below the edge
    areas = [livewire.auto_mask_array(t).sum() for t in (5, reach, 95)]
    assert areas[0] <= areas[1] <= areas[2], f"area not monotone in reach: {areas}"
    assert livewire.auto_mask_array(max(4, reach // 2)).sum() > 0.9 * disk.sum(), \
        "disk should already be full well below the auto reach"


def test_auto_select_two_tone_object_needs_two_seeds():
    """An object made of two flat colour halves on a background: one seed
    selects its half only (the internal edge is real); a second seed on
    the other half selects the whole object; a negative seed on the
    unselected half must not disturb a correct single-half selection."""
    h, w = 200, 300
    rgb = np.full((h, w, 3), (235, 235, 230), np.uint8)
    left = np.zeros((h, w), bool); left[50:150, 60:150] = True
    right = np.zeros((h, w), bool); right[50:150, 150:240] = True
    rgb[left] = (40, 80, 160)
    rgb[right] = (170, 60, 50)
    rng = np.random.default_rng(2)
    rgb = np.clip(rgb.astype(np.float32) + rng.normal(0, 2.0, rgb.shape), 0, 255).astype(np.uint8)
    livewire.set_image(rgba_from_rgb(rgb), w, h)

    reach = livewire.auto_select([100, 100], [])
    m1 = livewire.auto_mask_array(reach)
    print(f"  two-tone: one seed -> reach {reach}, IoU(left) {iou(m1, left):.3f}, IoU(both) {iou(m1, left | right):.3f}")
    assert iou(m1, left) > 0.95, "single seed should select exactly its half"

    livewire.auto_select([100, 100, 195, 100], [])
    m2 = livewire.auto_mask_array(reach)
    print(f"  two-tone: two seeds -> IoU(both) {iou(m2, left | right):.3f}")
    assert iou(m2, left | right) > 0.95, "two seeds should cover both halves"

    livewire.auto_select([100, 100], [200, 130])
    m3 = livewire.auto_mask_array(reach)
    print(f"  two-tone: left seed + negative on the right -> IoU(left) {iou(m3, left):.3f}")
    assert iou(m3, left) > 0.95, "a negative seed beyond an edge must not eat into the selection"


def test_auto_select_leak_is_stopped_by_negative_seed():
    """A disk joined to a second blob by a thin bridge of the same colour:
    a single seed leaks through the bridge (as it should — there is no
    edge), a negative seed on the second blob cuts the leak off."""
    h, w = 220, 360
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.hypot(xx - 100, yy - 110) < 60
    b = np.hypot(xx - 270, yy - 110) < 55
    bridge = (np.abs(yy - 110) < 4) & (xx > 100) & (xx < 270)
    obj = a | b | bridge
    rgb = np.where(obj[..., None], np.array([50, 160, 90], np.uint8), np.array([240, 240, 235], np.uint8))
    livewire.set_image(rgba_from_rgb(rgb), w, h)

    reach = livewire.auto_select([100, 110], [])
    leaked = livewire.auto_mask_array(reach)
    print(f"  leak: single seed reach {reach} -> IoU(a) {iou(leaked, a):.3f}, IoU(a|b) {iou(leaked, obj):.3f}")
    assert iou(leaked, obj) > 0.9, "the bridge should let the flood through (no edge to stop it)"

    livewire.auto_select([100, 110], [270, 110])
    fixed = livewire.auto_mask_array(reach)
    print(f"  leak: + negative seed on b -> IoU(a) {iou(fixed, a):.3f}, b left {int((fixed & b).sum())} px")
    assert iou(fixed, a) > 0.85, "negative seed should keep only disk a"
    assert (fixed & b).sum() < 0.05 * b.sum(), "negative seed's blob should be gone"


def test_auto_mask_contours_keep_holes_and_fill_specks():
    """A ring keeps its hole as a second loop; a tiny speck inside a disk
    gets filled; the contour polygons reproduce the mask area exactly."""
    h, w = 200, 200
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - 100, yy - 100)
    ring = (rr < 70) & (rr > 30)
    rgb = np.where(ring[..., None], np.array([200, 80, 40], np.uint8), np.array([30, 30, 35], np.uint8))
    livewire.set_image(rgba_from_rgb(rgb), w, h)
    reach = livewire.auto_select([100, 40], [])
    buf, lens, area = livewire.auto_mask(reach)
    mask = livewire.auto_mask_array(reach)
    print(f"  ring: reach {reach}, {len(lens)} loops, IoU {iou(mask, ring):.3f}")
    assert iou(mask, ring) > 0.95, "ring not selected"
    assert len(lens) == 2, f"ring should trace 2 loops (outer + hole), got {len(lens)}"
    coords = np.frombuffer(buf, np.float32).reshape(-1, 2)
    assert abs(polygon_area(coords, lens) - area) < 1, "loops (with hole) must reproduce the mask area"

    disk = rr < 70
    rgb2 = np.where(disk[..., None], np.array([200, 80, 40], np.uint8), np.array([30, 30, 35], np.uint8))
    rgb2[98:101, 98:101] = (30, 30, 35)                      # 3x3 speck of background colour
    livewire.set_image(rgba_from_rgb(rgb2), w, h)
    reach = livewire.auto_select([60, 100], [])
    buf, lens, area = livewire.auto_mask(reach)
    print(f"  speck: reach {reach}, {len(lens)} loop(s), area {area} (disk {disk.sum()})")
    assert len(lens) == 1, "a tiny interior speck should be filled, not traced as a hole"
    assert area >= 0.97 * disk.sum(), "disk lost pixels (boundary may sit <1px inside the blurred edge)"


def test_trace_contours_handles_saddles_and_multiple_blobs():
    """Direct contour-tracer check on a mask with a diagonal 'saddle'
    (two pixels touching at a corner), two separate blobs and a hole."""
    m = np.zeros((12, 14), bool)
    m[1:4, 1:4] = True; m[3, 3] = True; m[4, 4] = True; m[4:7, 4:7] = True   # blob w/ saddle at (4,4)
    m[8:11, 2:8] = True; m[9, 4:6] = False                                   # blob with a hole
    m[1:3, 10:13] = True                                                     # third blob
    coords, lens = livewire._trace_contours(m)
    total = polygon_area(coords, lens)
    print(f"  tracer: {len(lens)} loops, polygon area {total:.0f}, mask area {m.sum()}")
    assert abs(total - m.sum()) < 1e-6, "polygons must reproduce the mask area"
    assert len(lens) == 4, f"expected 4 loops (3 blobs + 1 hole), got {len(lens)}"
    assert coords.min() >= 0 and coords[:, 0].max() <= 14 and coords[:, 1].max() <= 12


def test_auto_select_smart_mode_sees_past_texture_and_creases():
    """Smart mode learns the object's appearance from the seeds, so
    (1) a noisily textured disk on a differently coloured, equally
    textured background is selected with one click at the auto reach,
    and (2) a shaded disk cut into cells by dark creases is selected
    whole from a short stroke — where "edges" mode has to stop at
    every crease (its auto reach fills a single cell)."""
    from scipy import ndimage
    size = 320
    yy, xx = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(5)

    disk = np.hypot(xx - 160, yy - 150) < 90
    inside = np.array([70, 130, 80.0]) + rng.normal(0, 28, (size, size, 3))
    outside = np.array([120, 100, 150.0]) + rng.normal(0, 28, (size, size, 3))
    rgb = np.clip(np.where(disk[..., None], inside, outside), 0, 255).astype(np.uint8)
    livewire.set_image(rgba_from_rgb(rgb), size, size)
    reach = livewire.auto_select([160, 150], [], "smart")
    smart = livewire.auto_mask_array(reach)
    print(f"  textured disk: smart reach {reach}, IoU {iou(smart, disk):.3f}")
    assert iou(smart, disk) > 0.9, "smart mode should select the whole textured disk"
    assert reach < 60, f"texture should be cheap to cross in smart mode (reach {reach})"

    disk2 = np.hypot(xx - 160, yy - 160) < 110
    creases = ((xx % 40) < 3) | ((yy % 40) < 3)
    shade = 1.0 - 0.5 * ((xx - 50) / 220.0)
    rgb = np.where(disk2[..., None], np.array([40, 90, 200.0]) * shade[..., None], np.array([210, 215, 225.0]))
    rgb[disk2 & creases] *= 0.35
    rgb = np.clip(rgb + rng.normal(0, 2, rgb.shape), 0, 255).astype(np.uint8)
    livewire.set_image(rgba_from_rgb(rgb), size, size)
    stroke = [v for i in range(7) for v in (100 + i * 20, 160)]   # across 3-4 cells

    reach_e = livewire.auto_select([160, 160], [], "edges")
    cell = livewire.auto_mask_array(reach_e)
    reach_s = livewire.auto_select(stroke, [], "smart")
    whole = livewire.auto_mask_array(reach_s)
    print(f"  creased disk: edges click -> reach {reach_e}, IoU {iou(cell, disk2):.3f} (one cell); "
          f"smart stroke -> reach {reach_s}, IoU {iou(whole, disk2):.3f}")
    assert iou(cell, disk2) < 0.3, "edges mode is expected to stop at the first crease"
    assert iou(whole, disk2) > 0.9, "smart mode should fill the whole creased disk from a stroke"
    assert (whole & ~disk2).sum() < 0.03 * disk2.sum(), "smart mode leaked into the background"

    # the mode is part of the cache key: switching back must recompute
    reach_e2 = livewire.auto_select(stroke, [], "edges")
    assert livewire._auto_key == (tuple(map(float, stroke)), (), "edges")


def bench_auto_select_browser_resolution():
    """Auto-select timing at the app's working resolution: graph build
    (once per image), a flood (per seed change), a re-threshold (per
    slider move). The browser (WASM) is roughly 2-3x slower."""
    w, h = 768, 576
    rng = np.random.default_rng(9)
    base = ndimage_blur(rng.random((h, w, 3)).astype(np.float32) * 255)
    rgb = np.clip(base, 0, 255).astype(np.uint8)
    livewire.set_image(rgba_from_rgb(rgb), w, h)
    t0 = time.perf_counter()
    livewire._ensure_auto_graph()
    t1 = time.perf_counter()
    reach = livewire.auto_select([w // 2, h // 2], [], "edges")
    t2 = time.perf_counter()
    livewire.auto_mask(reach)
    t3 = time.perf_counter()
    livewire.auto_mask(max(1, reach - 5))
    t4 = time.perf_counter()
    livewire.auto_select([w // 2, h // 2], [], "smart")
    t5 = time.perf_counter()
    print(f"  bench auto {w}x{h}: graph {t1 - t0:.2f}s | edge flood {t2 - t1:.2f}s | "
          f"smart (edge flood + model + flood) {t5 - t4:.2f}s | "
          f"mask+contours {(t3 - t2) * 1000:.0f}ms | re-threshold {(t4 - t3) * 1000:.0f}ms")


def ndimage_blur(a):
    from scipy import ndimage
    sig = (3.0,) * a.ndim if a.ndim == 2 else (3.0, 3.0, 0.0)
    return ndimage.gaussian_filter(a, sig)


if __name__ == "__main__":
    print("test_path_clings_to_disk_outline")
    test_path_clings_to_disk_outline()
    print("test_flat_image_gives_straight_path")
    test_flat_image_gives_straight_path()
    print("test_trim_cutout_removes_background_and_specks")
    test_trim_cutout_removes_background_and_specks()
    print("test_smooth_edges_rounds_jaggies_without_fringe")
    test_smooth_edges_rounds_jaggies_without_fringe()
    print("test_auto_select_disk_single_click")
    test_auto_select_disk_single_click()
    print("test_auto_select_two_tone_object_needs_two_seeds")
    test_auto_select_two_tone_object_needs_two_seeds()
    print("test_auto_select_leak_is_stopped_by_negative_seed")
    test_auto_select_leak_is_stopped_by_negative_seed()
    print("test_auto_mask_contours_keep_holes_and_fill_specks")
    test_auto_mask_contours_keep_holes_and_fill_specks()
    print("test_trace_contours_handles_saddles_and_multiple_blobs")
    test_trace_contours_handles_saddles_and_multiple_blobs()
    print("test_auto_select_smart_mode_sees_past_texture_and_creases")
    test_auto_select_smart_mode_sees_past_texture_and_creases()
    print("bench_browser_resolution")
    bench_browser_resolution()
    print("bench_auto_select_browser_resolution")
    bench_auto_select_browser_resolution()
    print("OK — all livewire tests passed")
