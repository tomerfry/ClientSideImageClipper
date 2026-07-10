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
3. **Cut** — close the path (double-click / `Enter` / click the first
   anchor) to get an antialiased, transparent-background PNG you can
   download or copy straight to the clipboard.
4. **Auto-trim** — tap `Space` (or the *trim bg* button): background-
   coloured areas reachable from the cutout's outer edge become
   transparent, stray disconnected specks are dropped, and the canvas is
   cropped tight to the subject (scipy connected-component labelling; the
   tolerance slider controls how aggressive the colour match is; every
   trim re-derives from the original cut, so it never compounds).

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

The engine runs in a Web Worker so the UI never blocks; images are
processed at a max working resolution of 960 px (path coordinates are
mapped back to full resolution for the final cut).

## Controls

| action | input |
|---|---|
| add snapping anchor | click |
| straight (non-snapping) segment | `Alt` + click |
| close path & cut | double-click / `Enter` / click first anchor |
| auto-trim leftover background | `Space` (tap) or *trim bg* button |
| undo last anchor | `Backspace` |
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
livewire.py   the path engine (runs identically under CPython and Pyodide)
tests/        native sanity tests + benchmark for livewire.py
```
