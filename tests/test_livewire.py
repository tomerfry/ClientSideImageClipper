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


def ndimage_blur(a):
    from scipy import ndimage
    return ndimage.gaussian_filter(a, 3.0)


if __name__ == "__main__":
    print("test_path_clings_to_disk_outline")
    test_path_clings_to_disk_outline()
    print("test_flat_image_gives_straight_path")
    test_flat_image_gives_straight_path()
    print("bench_browser_resolution")
    bench_browser_resolution()
    print("OK — all livewire tests passed")
