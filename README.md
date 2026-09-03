# › magic-clipper

Content-aware ("magnetic lasso") image clipping that runs **entirely in the
browser** — the computer-vision core is real Python (numpy + scipy) executed
client-side via [Pyodide](https://pyodide.org) (CPython compiled to
WebAssembly). No server, no uploads: a static site deployable to GitHub Pages.

## What it does

1. **Import an image** — open a file, drag & drop, or paste (`Ctrl+V`).
2. **Magic clipping** — click along an object's outline; the live path
   *clings* to edges as you move the cursor (intelligent-scissors /
   livewire, the algorithm behind Photoshop's Magnetic Lasso).
3. **Auto-select** — `Shift`+click an object and its boundary is
   detected for you: a flood spreads out from the click and stops
   wherever it would have to cross a colour edge. `Shift`+drag paints a
   stroke of seeds (do this across multi-part objects: each fold, petal
   or panel you touch gets filled to *its* edges), `Shift`+right-click
   (or `Shift`+`Alt`+click) subtracts a region that leaked in, and the
   **reach** slider / `[` `]` keys grow or shrink the result — the first
   click picks a reach automatically. Holes are kept.
4. **Cut** — close the path (double-click / `Enter` / click the first
   anchor) or press `Enter` on an auto-selection to get an antialiased,
   transparent-background PNG you can download or copy straight to the
   clipboard.
5. **Auto-trim** — press `B` (or the *trim bg* button): background-
   coloured areas reachable from the outer edge become transparent, stray
   disconnected specks are dropped, and the canvas is cropped tight to
   the subject (scipy connected-component labelling; the tolerance slider
   controls how aggressive the colour match is; every trim re-derives
   from its source, so it never compounds). Works on a finished cut — or
   directly on the imported image if you haven't cut anything yet, which
   is handy for cleaning up pasted cutouts that still carry junk.
6. **Smooth outline** — press `S` (or drag the *smooth* slider): rounds
   jagged, stair-stepped cutout boundaries while keeping a crisp ~1px
   anti-aliased edge. The object's colours are first extended outward
   (distance transform), so the reshaped edge never shows a dark fringe.
   Slider position is a live, non-destructive setting — 0 restores the
   original edge.

## How the "magic" works

`livewire.py` (Mortensen & Barrett, *Intelligent Scissors for Image
Composition*, SIGGRAPH '95):

- A per-pixel **cost map**: cheap on outlines (high Sobel gradient
  magnitude + Laplacian zero-crossings), expensive on flat regions.
- The image becomes an **8-connected graph**; dropping an anchor runs
  **Dijkstra** (scipy's C implementation) from that seed over the whole
  grid and keeps the predecessor tree.
- Every cursor move then extracts the globally-optimal seed→cursor path
  with a trivial predecessor walk — that's why the snapping feels instant.

**Auto-select** (`Shift`+click) is a *geodesic barrier flood* on the
same grid, with the costs turned inside out:

- Stepping onto a pixel costs its **colour-gradient magnitude** (Sobel
  over the three channels, scaled so a step of *h* grey levels costs
  about *h* however blurry it is — the gradient integrates to the
  contrast). A soft noise floor keeps texture and noise cheap without
  letting wide, blurry outlines leak.
- Dijkstra from the seed pixel(s) (`min_only` multi-source) then gives
  every pixel its **minimum crossing cost** — "how much edge do I have
  to cross to get here". Flat and shaded interiors cost ~nothing, real
  outlines cost their contrast, so the region fills the object and
  stops at its edge.
- The selection is the sublevel set `d(p) ≤ reach`. Re-thresholding is
  free, so the slider adjusts the result without re-running Dijkstra;
  the initial reach is chosen automatically from the region-growth
  curve (the first wide *plateau* of area-vs-reach — the object is
  full, the flood hasn't leaked yet — cut through its middle so soft
  edges are split down the ramp).
- Negative seeds run a second flood; a pixel stays selected only if it
  is geodesically closer to a positive seed than to a negative one
  (the GeoS rule), which lets one click push back a leak.
- The mask is tidied (seed-connected components, 1-px cracks closed,
  small holes filled) and turned into **pixel-corner polygons** by a
  vectorised crack-following tracer — holes come out as their own
  loops, so an even-odd fill reproduces the mask exactly, and the same
  polygon pipeline (Chaikin smoothing, antialiased canvas fill) does the
  cut at full resolution.

Works best on objects with a visible outline; strongly textured
subjects (foliage, fabric weave) accumulate cost and need the lasso.

The engine runs in a Web Worker so the UI never blocks; images are
processed at a max working resolution of 960 px (path coordinates are
mapped back to full resolution for the final cut).

## Controls

| action | input |
|---|---|
| add snapping anchor | click |
| straight (non-snapping) segment | `Alt` + click |
| auto-select the object under the cursor | `Shift` + click, `Shift` + drag to paint seeds |
| subtract from the auto-selection | `Shift` + right-click / `Shift` + `Alt` + click |
| grow / shrink the auto-selection | *reach* slider or `[` / `]` |
| close path & cut | double-click / `Enter` / click first anchor |
| auto-trim leftover background | `B` or *trim bg* button |
| smooth cutout outline | `S` or the *smooth* slider |
| undo last anchor / seed stroke | `Backspace` |
| discard path | `Esc` |
| zoom / pan | wheel / `Space`+drag or middle-drag |
| fit to view | `F` |

## Run locally

Any static file server works (a worker + `fetch` need http, not `file://`):

```sh
python -m http.server 8000
# open http://localhost:8000
```

Test the Python engine natively (needs numpy + scipy):

```sh
python tests/test_livewire.py
```

## Deploy to GitHub Pages

1. Push this repo to GitHub (default branch `main`).
2. In the repo: **Settings → Pages → Build and deployment → Source:
   GitHub Actions**.
3. Push to `main` (or run the `deploy-pages` workflow manually) — the site
   appears at `https://<user>.github.io/<repo>/`.

The workflow (`.github/workflows/deploy.yml`) also runs the engine tests
on every deploy, so a broken algorithm never ships.

## Files

```
index.html    UI shell (Monokai-themed)
app.js        editor: view transform, lasso state, rendering, PNG cutout
worker.js     Web Worker hosting Pyodide + numpy/scipy
livewire.py   the path + auto-select engine (runs identically under CPython and Pyodide)
tests/        native sanity tests + benchmark for livewire.py
```
