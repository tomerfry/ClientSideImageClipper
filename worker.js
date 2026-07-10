/* Web Worker: boots Pyodide (CPython compiled to WebAssembly), loads
 * numpy + scipy, imports livewire.py, and serves the UI thread:
 *
 *   init      -> ready            (engine booted)
 *   setImage  -> imageReady       (cost graph built)
 *   seed      -> seedReady        (Dijkstra tree computed for an anchor)
 *   path      -> path             (seed->cursor optimal path, Int32Array)
 *
 * Messages are handled strictly in order, so a `path` reply always
 * reflects the most recent `seed`/`setImage` that preceded it.
 */

'use strict';

const PYODIDE_BASE = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/';
importScripts(PYODIDE_BASE + 'pyodide.js');

let livewire = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const fail = (context, err) =>
  post({ type: 'error', context, text: (err && err.message) || String(err) });

async function init(pySourceUrl) {
  try {
    post({ type: 'status', text: 'loading python runtime (pyodide wasm)…' });
    const pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });

    post({ type: 'status', text: 'loading numpy + scipy…' });
    await pyodide.loadPackage(['numpy', 'scipy'], {
      messageCallback: (m) => post({ type: 'status', text: String(m).toLowerCase() }),
    });

    post({ type: 'status', text: 'importing livewire engine…' });
    const src = await (await fetch(pySourceUrl)).text();
    pyodide.FS.writeFile('/home/pyodide/livewire.py', src);
    pyodide.runPython(
      'import sys\n' +
      'if "/home/pyodide" not in sys.path:\n' +
      '    sys.path.insert(0, "/home/pyodide")\n'
    );
    livewire = pyodide.pyimport('livewire');
    post({ type: 'ready' });
  } catch (err) {
    fail('init', err);
  }
}

self.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'init':
      init(m.pySourceUrl);
      break;

    case 'setImage':
      try {
        livewire.set_image(m.rgba, m.width, m.height);
        post({ type: 'imageReady', width: m.width, height: m.height });
      } catch (err) {
        fail('setImage', err);
      }
      break;

    case 'seed':
      try {
        livewire.set_seed(m.x, m.y);
        post({ type: 'seedReady', gen: m.gen });
      } catch (err) {
        fail('seed', err);
      }
      break;

    case 'trim':
      try {
        const proxy = livewire.trim_cutout(m.rgba, m.width, m.height, m.tolerance);
        const [buf, x, y, width, height] = proxy.toJs();
        proxy.destroy();
        const rgba = new Uint8ClampedArray(buf && buf.length ? buf : 0);
        post({ type: 'trimmed', rgba, width, height, x, y }, [rgba.buffer]);
      } catch (err) {
        fail('trim', err);
      }
      break;

    case 'smooth':
      try {
        const proxy = livewire.smooth_edges(m.rgba, m.width, m.height, m.amount);
        const [buf, x, y, width, height] = proxy.toJs();
        proxy.destroy();
        const rgba = new Uint8ClampedArray(buf && buf.length ? buf : 0);
        post({ type: 'smoothed', rgba, width, height, x, y }, [rgba.buffer]);
      } catch (err) {
        fail('smooth', err);
      }
      break;

    case 'path':
      try {
        const proxy = livewire.get_path(m.x, m.y);
        const list = proxy.toJs();
        proxy.destroy();
        const points = Int32Array.from(list);
        post({ type: 'path', token: m.token, points }, [points.buffer]);
      } catch (err) {
        fail('path', err);
      }
      break;
  }
};
