/* magic-clipper UI thread.
 *
 * Owns: image loading (open/drop/paste), the pan/zoom view transform,
 * magnetic-lasso path state, auto-select (shift+click) seed state, canvas
 * rendering, and cutout generation. The engine itself (numpy/scipy
 * Dijkstra) lives in worker.js behind init / setImage / seed / path plus
 * auto / autoReach for the click-to-object flood.
 *
 * Two ways to build a selection, never both at once:
 *   lasso  — anchors + snapped segments (one closed polygon)
 *   auto   — shift+click / shift+drag seeds -> engine returns the
 *            object's boundary loops (holes included); the `reach`
 *            slider re-thresholds the same flood instantly; `smart`
 *            mode adds a colour+texture model on top of the edges
 * Both end up as `selectionPolygons()` (full-image px, even-odd fill).
 *
 * Coordinate spaces:
 *   screen px (css)  =  full-image px * view.scale + view.t
 *   full-image px    =  (work px + 0.5) * (wsx, wsy)
 * The engine only ever sees integer *work* coordinates (image downscaled
 * to WORK_MAX on its longest side, keeping Dijkstra interactive).
 */

'use strict';

const WORK_MAX = 960;          // engine grid: longest image side, px
const CLOSE_RADIUS_PX = 12;    // screen px: click near first anchor closes
const DUP_RADIUS_PX = 5;       // screen px: clicks this close to the last anchor are ignored
const REACH_MAX = 100;         // auto-select slider range (engine contrast units)
const STROKE_STEP_PX = 6;      // screen px between seeds while shift+dragging

const $ = (id) => document.getElementById(id);
const els = {
  canvas: $('canvas'), stage: $('stage'), dropzone: $('dropzone'),
  bootlog: $('bootlog'), bootLines: $('boot-lines'),
  busy: $('busy'), busyText: $('busy-text'),
  dot: $('dot'), statusText: $('status-text'), imgInfo: $('img-info'),
  fileInput: $('file-input'),
  resultImg: $('result-img'), resultPlaceholder: $('result-placeholder'), resultMeta: $('result-meta'),
  open: $('btn-open'), paste: $('btn-paste'), undo: $('btn-undo'), reset: $('btn-reset'),
  cut: $('btn-cut'), fit: $('btn-fit'), download: $('btn-download'), copy: $('btn-copy'),
  trim: $('btn-trim'), tol: $('tol'), tolVal: $('tol-val'),
  smooth: $('smooth'), smoothVal: $('smooth-val'),
  reach: $('reach'), reachVal: $('reach-val'), reachMode: $('reach-mode'),
  autoMode: $('auto-mode'),
};
const ctx = els.canvas.getContext('2d');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const state = {
  bitmap: null, fullW: 0, fullH: 0,
  workW: 0, workH: 0, wsx: 1, wsy: 1,
  view: { scale: 1, tx: 0, ty: 0 },

  engineReady: false, imageReady: false,
  pendingImage: null,          // queued setImage payload while engine boots
  graphBusy: false, seedBusy: false,
  seedGen: 0,                  // bumped per seed; stale path replies are dropped
  inFlight: false,             // one path request at a time; latest wins
  wantPath: null,              // {x, y} latest cursor target awaiting a request
  pendingCommit: null,         // {x, y, close} to commit on its path reply

  anchors: [],                 // [{x, y}] work coords
  segments: [],                // [Int32Array flat x,y work coords], segments[i] ends at anchors[i+1]
  livePath: null,              // Int32Array, current seed -> cursor
  closed: false,

  auto: null,                  // auto-select session (see newAuto)
  autoGen: 0,                  // bumped per auto request; stale replies are dropped
  stroke: null,                // shift+drag in progress: {sign, group, sx, sy}
  shiftDown: false,
  selPath: null,               // cached {sel, dim} Path2D of the selection (full-image px)

  cutout: null,                // {blob, url, w, h, srcCanvas, baseCanvas, trimmed, smoothed}
  trimSrc: null,               // canvas the in-flight/last trim derives from
  smoothBase: null,            // canvas the in-flight smooth derives from
  trimBusy: false, smoothBusy: false,
  spaceDown: false, pan: null,
};

/* ── worker wiring ─────────────────────────────────────────────── */

const worker = new Worker('worker.js');
worker.postMessage({ type: 'init', pySourceUrl: new URL('livewire.py', location.href).href });

worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'status':
      bootLine(m.text);
      setStatus('loading', m.text);
      break;

    case 'ready':
      state.engineReady = true;
      els.bootlog.classList.add('hidden');
      setStatus('ready', 'python engine ready — open, drop, or paste an image');
      if (state.pendingImage) {
        const p = state.pendingImage;
        state.pendingImage = null;
        postImage(p);
      }
      break;

    case 'imageReady':
      state.graphBusy = false;
      state.imageReady = true;
      hideBusy();
      setStatus('ready', 'ready — click an outline to clip, or press b to auto-trim the whole image');
      updateUi();
      break;

    case 'seedReady':
      if (m.gen !== state.seedGen) break; // superseded by a newer anchor
      state.seedBusy = false;
      hideBusy();
      pumpPath();
      break;

    case 'path':
      onPathReply(m.token, m.points);
      break;

    case 'busy':
      showBusy(m.text);
      break;

    case 'auto':
      onAutoResult(m);
      break;

    case 'trimmed':
      state.trimBusy = false;
      hideBusy();
      if (!state.trimSrc) break;
      if (!m.width || !m.height) {
        setStatus('ready', 'trim removed everything — lower the tolerance and retry');
        break;
      }
      applyTrimmed(m);
      break;

    case 'smoothed':
      state.smoothBusy = false;
      hideBusy();
      if (!state.smoothBase) break;
      if (!m.width || !m.height) {
        setStatus('ready', 'smoothing removed everything — lower the smooth value');
        break;
      }
      applySmoothed(m);
      break;

    case 'error':
      console.error('worker error in', m.context, m.text);
      setStatus('error', `python engine error (${m.context}): ${m.text}`);
      hideBusy();
      if ((m.context === 'auto' || m.context === 'autoReach') && state.auto) {
        state.auto.inFlight = false;
        state.auto.pending = null;
        state.auto.closeWhenIdle = false;
      }
      break;
  }
};

function sendSeed(pt) {
  state.seedGen++;
  state.seedBusy = true;
  state.livePath = null;
  showBusy('snapping to edges…');
  worker.postMessage({ type: 'seed', x: pt.x, y: pt.y, gen: state.seedGen });
}

function sendPath(x, y, purpose) {
  state.inFlight = true;
  worker.postMessage({ type: 'path', x, y, token: { gen: state.seedGen, purpose, x, y } });
}

function pumpPath() {
  if (state.inFlight || !state.wantPath) return;
  if (!canTrace()) return;
  const p = state.wantPath;
  state.wantPath = null;
  sendPath(p.x, p.y, 'live');
}

function canTrace() {
  return state.imageReady && state.anchors.length > 0 && !state.closed &&
         !state.seedBusy && !state.pendingCommit;
}

function onPathReply(token, points) {
  state.inFlight = false;
  if (token.gen !== state.seedGen) { pumpPath(); return; } // stale seed
  if (token.purpose === 'live') {
    state.livePath = points;
    requestRender();
    pumpPath();
    return;
  }
  // commit or close
  const pc = state.pendingCommit;
  if (!pc || pc.x !== token.x || pc.y !== token.y) { pumpPath(); return; }
  state.pendingCommit = null;
  state.segments.push(points);
  state.livePath = null;
  invalidateSelection();
  if (token.purpose === 'close') {
    state.closed = true;
    hideBusy();
    buildCutout();
  } else {
    state.anchors.push({ x: token.x, y: token.y });
    sendSeed({ x: token.x, y: token.y });
  }
  updateUi();
  requestRender();
}

/* ── status / chrome helpers ───────────────────────────────────── */

function setStatus(kind, text) {
  els.dot.className = kind === 'ready' ? 'ready' : kind === 'error' ? 'error' : kind === 'busy' ? 'busy' : '';
  els.statusText.textContent = text;
}

function bootLine(text) {
  const div = document.createElement('div');
  div.textContent = text;
  els.bootLines.appendChild(div);
  while (els.bootLines.children.length > 4) els.bootLines.firstChild.remove();
}

function showBusy(text) {
  els.busyText.textContent = text;
  els.busy.classList.add('on');
  setStatus('busy', text);
}

function hideBusy() {
  els.busy.classList.remove('on');
  if (state.engineReady) setStatus('ready', state.imageReady ? 'ready' : 'python engine ready');
}

function updateUi() {
  const autoSel = !!(state.auto && state.auto.contours.length);
  const hasPath = state.anchors.length > 0 || !!state.auto;
  els.undo.disabled = !hasPath;
  els.reset.disabled = !hasPath;
  els.cut.disabled = !((state.anchors.length >= 2 || autoSel) && !state.closed);
  els.fit.disabled = !state.bitmap;
  els.reach.disabled = !state.imageReady;
  els.autoMode.disabled = !state.imageReady;
  els.download.disabled = !state.cutout;
  els.copy.disabled = !state.cutout || typeof ClipboardItem === 'undefined';
  els.trim.disabled = !state.engineReady || (!state.cutout && !state.bitmap);
}

/* ── image loading ─────────────────────────────────────────────── */

async function loadImageBlob(blob) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  } catch {
    try {
      bitmap = await createImageBitmap(blob);
    } catch (err) {
      setStatus('error', `could not decode image: ${err.message || err}`);
      return;
    }
  }
  resetPath();
  clearCutout();
  state.bitmap = bitmap;
  state.fullW = bitmap.width;
  state.fullH = bitmap.height;
  state.imageReady = false;

  const down = Math.max(1, Math.max(state.fullW, state.fullH) / WORK_MAX);
  state.workW = Math.max(2, Math.round(state.fullW / down));
  state.workH = Math.max(2, Math.round(state.fullH / down));
  state.wsx = state.fullW / state.workW;
  state.wsy = state.fullH / state.workH;

  const oc = document.createElement('canvas');
  oc.width = state.workW;
  oc.height = state.workH;
  oc.getContext('2d').drawImage(bitmap, 0, 0, state.workW, state.workH);
  const rgba = oc.getContext('2d').getImageData(0, 0, state.workW, state.workH).data;

  els.dropzone.classList.add('hidden');
  els.imgInfo.textContent = `${state.fullW}×${state.fullH}px · snap grid ${state.workW}×${state.workH}`;
  fitView();
  updateUi();
  requestRender();

  const payload = { rgba, w: state.workW, h: state.workH };
  if (state.engineReady) postImage(payload);
  else {
    state.pendingImage = payload;
    showBusy('waiting for python engine…');
  }
}

function clearCutout() {
  if (state.cutout) URL.revokeObjectURL(state.cutout.url);
  state.cutout = null;
  state.trimSrc = null;
  state.smoothBase = null;
  els.resultImg.style.display = 'none';
  els.resultImg.removeAttribute('src');
  els.resultPlaceholder.style.display = 'block';
  els.resultMeta.textContent = '';
  updateUi();
}

function postImage(p) {
  state.graphBusy = true;
  showBusy('building edge cost graph…');
  worker.postMessage(
    { type: 'setImage', rgba: p.rgba, width: p.w, height: p.h },
    [p.rgba.buffer]
  );
}

els.open.addEventListener('click', () => els.fileInput.click());
els.fileInput.addEventListener('change', () => {
  if (els.fileInput.files[0]) loadImageBlob(els.fileInput.files[0]);
  els.fileInput.value = '';
});

window.addEventListener('paste', (e) => {
  for (const item of e.clipboardData?.items || []) {
    if (item.type.startsWith('image/')) {
      e.preventDefault();
      loadImageBlob(item.getAsFile());
      return;
    }
  }
});

els.paste.addEventListener('click', async () => {
  try {
    for (const item of await navigator.clipboard.read()) {
      const type = item.types.find((t) => t.startsWith('image/'));
      if (type) { loadImageBlob(await item.getType(type)); return; }
    }
    setStatus('ready', 'no image on the clipboard');
  } catch {
    setStatus('ready', 'clipboard blocked — press ctrl+v instead');
  }
});

['dragover', 'dragenter'].forEach((ev) =>
  window.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove('hidden');
    els.dropzone.classList.add('armed');
  })
);
window.addEventListener('dragleave', (e) => {
  if (e.relatedTarget) return;
  els.dropzone.classList.remove('armed');
  if (state.bitmap) els.dropzone.classList.add('hidden');
});
window.addEventListener('drop', (e) => {
  e.preventDefault();
  els.dropzone.classList.remove('armed');
  if (state.bitmap) els.dropzone.classList.add('hidden');
  const file = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith('image/'));
  if (file) loadImageBlob(file);
});

/* ── view transform (zoom / pan) ───────────────────────────────── */

function fitView() {
  const w = els.canvas.clientWidth, h = els.canvas.clientHeight;
  if (!state.bitmap || !w || !h) return;
  const scale = 0.97 * Math.min(w / state.fullW, h / state.fullH);
  state.view.scale = scale;
  state.view.tx = (w - state.fullW * scale) / 2;
  state.view.ty = (h - state.fullH * scale) / 2;
  requestRender();
}

els.canvas.addEventListener('wheel', (e) => {
  if (!state.bitmap) return;
  e.preventDefault();
  const r = els.canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const v = state.view;
  const k = Math.exp(-e.deltaY * 0.0016);
  const next = Math.min(60, Math.max(0.02, v.scale * k));
  const applied = next / v.scale;
  v.tx = mx - (mx - v.tx) * applied;
  v.ty = my - (my - v.ty) * applied;
  v.scale = next;
  requestRender();
}, { passive: false });

/* ── pointer input ─────────────────────────────────────────────── */

function eventToWork(e) {
  const r = els.canvas.getBoundingClientRect();
  const fx = (e.clientX - r.left - state.view.tx) / state.view.scale;
  const fy = (e.clientY - r.top - state.view.ty) / state.view.scale;
  return {
    x: Math.min(state.workW - 1, Math.max(0, Math.round(fx / state.wsx - 0.5))),
    y: Math.min(state.workH - 1, Math.max(0, Math.round(fy / state.wsy - 0.5))),
  };
}

function workToScreen(pt) {
  return {
    x: (pt.x + 0.5) * state.wsx * state.view.scale + state.view.tx,
    y: (pt.y + 0.5) * state.wsy * state.view.scale + state.view.ty,
  };
}

function screenDist(e, workPt) {
  const r = els.canvas.getBoundingClientRect();
  const s = workToScreen(workPt);
  return Math.hypot(e.clientX - r.left - s.x, e.clientY - r.top - s.y);
}

els.canvas.addEventListener('pointerdown', (e) => {
  if (e.button === 1 || (e.button === 0 && state.spaceDown)) {
    e.preventDefault();
    state.pan = { sx: e.clientX, sy: e.clientY, tx: state.view.tx, ty: state.view.ty };
    els.canvas.classList.add('pan-active');
    els.canvas.setPointerCapture(e.pointerId);
    return;
  }
  if (!state.imageReady) return;
  if (e.shiftKey && (e.button === 0 || e.button === 2)) {
    e.preventDefault();
    beginAutoInput(e, e.button === 2 || e.altKey ? -1 : 1);
    return;
  }
  if (e.button !== 0 || state.closed) return;
  if (state.auto) {
    setStatus('ready', 'auto-select active — shift+click/drag adds, shift+right-click subtracts, enter cuts, esc clears');
    return;
  }

  const pt = eventToWork(e);
  const last = state.anchors[state.anchors.length - 1];
  if (last && screenDist(e, last) < DUP_RADIUS_PX) return; // debounce dbl-click's 2nd click

  if (state.anchors.length === 0) {
    state.anchors.push(pt);
    sendSeed(pt);
    updateUi();
    requestRender();
  } else if (state.anchors.length >= 2 && screenDist(e, state.anchors[0]) < CLOSE_RADIUS_PX) {
    requestClose();
  } else if (state.seedBusy || state.pendingCommit) {
    // engine still chewing the previous anchor; ignore rapid clicks
  } else if (e.altKey) {
    commitStraight(pt);
  } else {
    state.pendingCommit = { x: pt.x, y: pt.y, close: false };
    sendPath(pt.x, pt.y, 'commit');
  }
});

els.canvas.addEventListener('pointermove', (e) => {
  if (state.pan) {
    state.view.tx = state.pan.tx + (e.clientX - state.pan.sx);
    state.view.ty = state.pan.ty + (e.clientY - state.pan.sy);
    requestRender();
    return;
  }
  if (state.stroke) { extendStroke(e); return; }
  if (!state.imageReady || state.closed || state.anchors.length === 0) return;
  state.wantPath = eventToWork(e);
  pumpPath();
});

function endPointer(e) {
  if (state.pan) {
    state.pan = null;
    els.canvas.classList.remove('pan-active');
    try { els.canvas.releasePointerCapture(e.pointerId); } catch {}
  }
  if (state.stroke) {
    state.stroke = null;
    try { els.canvas.releasePointerCapture(e.pointerId); } catch {}
  }
}
els.canvas.addEventListener('pointerup', endPointer);
els.canvas.addEventListener('pointercancel', endPointer);

els.canvas.addEventListener('dblclick', (e) => {
  e.preventDefault();
  if (e.shiftKey) return; // shift+dblclick is just two seeds, never a close
  requestClose();
});
els.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

function commitStraight(pt) {
  const a = state.anchors[state.anchors.length - 1];
  state.segments.push(Int32Array.from([a.x, a.y, pt.x, pt.y]));
  state.anchors.push(pt);
  state.livePath = null;
  invalidateSelection();
  sendSeed(pt);
  updateUi();
  requestRender();
}

function requestClose() {
  if (state.auto) { commitAuto(); return; }
  if (state.closed || state.anchors.length < 2 || state.pendingCommit) return;
  if (state.seedBusy) return; // last anchor's tree not ready yet
  const first = state.anchors[0];
  state.pendingCommit = { x: first.x, y: first.y, close: true };
  showBusy('closing path…');
  sendPath(first.x, first.y, 'close');
}

function undoAnchor() {
  if (state.auto) { undoAutoSeed(); return; }
  if (state.closed) {           // reopen: drop only the closing segment
    state.segments.pop();
    state.closed = false;
  } else if (state.anchors.length > 1) {
    state.segments.pop();
    state.anchors.pop();
    sendSeed(state.anchors[state.anchors.length - 1]);
  } else if (state.anchors.length === 1) {
    state.anchors.pop();
    state.livePath = null;
    state.seedGen++;            // orphan any in-flight seed/path replies
    state.seedBusy = false;
    hideBusy();
  }
  state.pendingCommit = null;
  invalidateSelection();
  updateUi();
  requestRender();
}

function resetPath() {
  state.anchors = [];
  state.segments = [];
  state.livePath = null;
  state.closed = false;
  state.pendingCommit = null;
  state.wantPath = null;
  state.seedGen++;
  state.seedBusy = false;
  state.auto = null;
  state.stroke = null;
  state.autoGen++;            // orphan any in-flight auto reply
  invalidateSelection();
  setReachBadge();
  hideBusy();
  updateUi();
  requestRender();
}

els.undo.addEventListener('click', undoAnchor);
els.reset.addEventListener('click', resetPath);
els.cut.addEventListener('click', requestClose);
els.fit.addEventListener('click', fitView);

window.addEventListener('keydown', (e) => {
  const tag = e.target && e.target.tagName;
  if (tag === 'BUTTON' || tag === 'INPUT') return; // let native control keys work
  if (e.code === 'Space') {
    state.spaceDown = true;
    els.canvas.classList.add('panning');
    e.preventDefault();
  }
  else if (e.key === 'Shift') {
    state.shiftDown = true;
    els.canvas.classList.add('wand');
  }
  else if (e.key === '[') nudgeReach(-4);
  else if (e.key === ']') nudgeReach(4);
  else if (e.key === 'm' || e.key === 'M') toggleAutoMode();
  else if (e.key === 'b' || e.key === 'B') requestTrim();
  else if (e.key === 's' || e.key === 'S') {
    if (Number(els.smooth.value) === 0) {
      els.smooth.value = '4';               // sensible default on first press
      els.smoothVal.textContent = '4';
    }
    requestSmooth();
  }
  else if (e.key === 'Escape') resetPath();
  else if (e.key === 'Backspace' || e.key === 'Delete') { e.preventDefault(); undoAnchor(); }
  else if (e.key === 'Enter') requestClose();
  else if (e.key === 'f' || e.key === 'F') fitView();
  else if (e.key === 'o' || e.key === 'O') els.fileInput.click();
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space') {
    state.spaceDown = false;
    els.canvas.classList.remove('panning');
  }
  if (e.key === 'Shift') {
    state.shiftDown = false;
    els.canvas.classList.remove('wand');
  }
});
window.addEventListener('blur', () => { // modifier keys released while unfocused
  state.spaceDown = false;
  state.shiftDown = false;
  els.canvas.classList.remove('panning', 'wand');
});

/* ── auto-select (shift+click / shift+drag) ────────────────────── */

function newAuto() {
  return {
    seeds: [],          // [{x, y, sign, group}] work coords; a stroke shares one group
    group: 0,
    mode: els.autoMode.value,  // 'smart' (edges + appearance model) or 'edges' (outline only)
    contours: [],       // [Float32Array x,y corner coords in work px], holes included
    area: 0,            // work px inside the selection
    reach: null,        // null until the engine's first suggestion arrives
    suggested: null,
    manual: false,      // the user touched the reach slider for this selection
    inFlight: false,    // one engine request at a time; latest pending wins
    pending: null,      // {seeds?: true, reach?: n} merged until it can be sent
    closeWhenIdle: false, // enter/cut arrived while a request was in flight
    wasClosed: false,   // slider drag reopened a cut; re-cut on release
  };
}

function beginAutoInput(e, sign) {
  if (state.anchors.length) {
    setStatus('ready', state.closed
      ? 'a lasso cut is active — press esc to start over, then shift+click'
      : 'finish (enter) or discard (esc) the lasso before auto-selecting');
    return;
  }
  if (!state.auto) state.auto = newAuto();
  const a = state.auto;
  if (sign < 0 && !a.seeds.some((s) => s.sign > 0)) {
    setStatus('ready', 'shift+right-click subtracts — shift+click the object first');
    return;
  }
  const pt = eventToWork(e);
  const last = a.seeds[a.seeds.length - 1];
  if (last && last.sign === sign && last.x === pt.x && last.y === pt.y) return; // dblclick's 2nd click
  if (state.closed) reopenSelection();
  a.group++;
  a.seeds.push({ x: pt.x, y: pt.y, sign, group: a.group });
  state.stroke = { sign, group: a.group, sx: e.clientX, sy: e.clientY };
  try { els.canvas.setPointerCapture(e.pointerId); } catch {}
  requestAuto({ seeds: true });
  updateUi();
  requestRender();
}

function extendStroke(e) {
  const s = state.stroke, a = state.auto;
  if (!a) { state.stroke = null; return; }
  if (Math.hypot(e.clientX - s.sx, e.clientY - s.sy) < STROKE_STEP_PX) return;
  s.sx = e.clientX;
  s.sy = e.clientY;
  const pt = eventToWork(e);
  const last = a.seeds[a.seeds.length - 1];
  if (last && last.x === pt.x && last.y === pt.y) return;
  a.seeds.push({ x: pt.x, y: pt.y, sign: s.sign, group: s.group });
  requestAuto({ seeds: true });   // live preview while painting (latest wins)
  requestRender();
}

function requestAuto(change) {
  const a = state.auto;
  if (!a) return;
  a.pending = Object.assign(a.pending || {}, change);
  pumpAuto();
}

function pumpAuto() {
  const a = state.auto;
  if (!a || a.inFlight || !a.pending || !state.imageReady) return;
  const p = a.pending;
  a.pending = null;
  a.gen = ++state.autoGen;
  if (p.seeds) {
    const pos = [], neg = [];
    for (const s of a.seeds) (s.sign > 0 ? pos : neg).push(s.x, s.y);
    if (!pos.length) { a.contours = []; a.area = 0; invalidateSelection(); requestRender(); return; }
    a.inFlight = true;
    showBusy('detecting object…');
    // reach: null on the very first request -> the engine picks one
    worker.postMessage({ type: 'auto', pos, neg, mode: a.mode, reach: a.reach, gen: a.gen });
  } else {
    a.inFlight = true;
    worker.postMessage({ type: 'autoReach', reach: p.reach, gen: a.gen });
  }
}

function onAutoResult(m) {
  const a = state.auto;
  if (!a || m.gen !== a.gen) return; // stale (seeds changed / selection cleared meanwhile)
  a.inFlight = false;
  hideBusy();
  a.contours = splitContours(m.coords, m.lens);
  a.area = m.area;
  if (m.suggested != null) a.suggested = m.suggested;
  a.reach = m.reach;
  if (!a.manual) setReachUi(m.reach);
  setReachBadge();
  invalidateSelection();
  if (a.pending) pumpAuto();
  else if (a.closeWhenIdle) { a.closeWhenIdle = false; commitAuto(); }
  if (!state.closed) {
    const px = Math.round(a.area * state.wsx * state.wsy);
    setStatus('ready', a.area
      ? `object: ~${px.toLocaleString()} px · ${a.mode} · reach ${a.reach}${a.manual ? '' : ' (auto)'} — shift+drag adds, shift+right-click subtracts, [ ] reach, m mode, enter cuts`
      : 'nothing within reach — raise reach (]) or shift+click elsewhere');
  }
  updateUi();
  requestRender();
}

function splitContours(coords, lens) {
  const out = [];
  let o = 0;
  for (const n of lens) {
    out.push(coords.subarray(o, o + n * 2));
    o += n * 2;
  }
  return out;
}

function commitAuto() {
  const a = state.auto;
  if (state.closed) return;
  if (a.inFlight || a.pending) { a.closeWhenIdle = true; return; } // cut once the preview is current
  if (!a.contours.length) {
    setStatus('ready', 'nothing selected — shift+click or shift+drag over an object');
    return;
  }
  state.closed = true;
  a.wasClosed = false;
  invalidateSelection();
  buildCutout();
  updateUi();
  requestRender();
}

function reopenSelection() {
  state.closed = false;
  invalidateSelection();
  updateUi();
}

function undoAutoSeed() {
  const a = state.auto;
  if (state.closed) { reopenSelection(); requestRender(); return; }
  const g = a.seeds.length ? a.seeds[a.seeds.length - 1].group : 0;
  a.seeds = a.seeds.filter((s) => s.group !== g);   // a whole stroke at a time
  if (!a.seeds.length) { clearAuto(); return; }
  a.closeWhenIdle = false;
  requestAuto({ seeds: true });
  updateUi();
  requestRender();
}

function clearAuto() {
  state.auto = null;
  state.stroke = null;
  state.closed = false;
  state.autoGen++;
  invalidateSelection();
  setReachBadge();
  hideBusy();
  updateUi();
  requestRender();
}

function setReachUi(v) {
  els.reach.value = String(v);
  els.reachVal.textContent = String(v);
}

function setReachBadge() {
  const a = state.auto;
  els.reachMode.textContent = a && !a.manual && a.reach != null ? 'auto' : '';
}

function setReach(v) {
  v = Math.min(REACH_MAX, Math.max(1, Math.round(v)));
  setReachUi(v);
  const a = state.auto;
  if (!a) return;
  a.manual = true;
  a.reach = v;
  setReachBadge();
  if (state.closed) { reopenSelection(); a.wasClosed = true; } // preview edits; re-cut on release
  requestAuto({ reach: v });
}

function nudgeReach(delta) {
  if (!state.auto) {
    setStatus('ready', 'reach adjusts an auto-selection — shift+click an object first');
    return;
  }
  const recut = state.closed || state.auto.wasClosed;
  setReach(Number(els.reach.value) + delta);
  if (recut) { state.auto.wasClosed = false; requestClose(); }
}

function setAutoMode(mode) {
  els.autoMode.value = mode;
  const a = state.auto;
  if (!a || a.mode === mode) return;
  a.mode = mode;
  a.reach = null;           // the two modes' costs differ: let the engine re-pick the reach
  a.manual = false;
  setReachBadge();
  const wasClosed = state.closed;
  if (wasClosed) reopenSelection();
  requestAuto({ seeds: true });
  if (wasClosed) requestClose(); // re-cut once the new result is in
}

function toggleAutoMode() {
  setAutoMode(els.autoMode.value === 'smart' ? 'edges' : 'smart');
  if (!state.auto) setStatus('ready', `auto-select mode: ${els.autoMode.value} — shift+click an object`);
}

els.autoMode.addEventListener('change', () => setAutoMode(els.autoMode.value));
els.reach.addEventListener('input', () => setReach(Number(els.reach.value)));
els.reach.addEventListener('change', () => {
  const a = state.auto;
  if (a && a.wasClosed) { a.wasClosed = false; requestClose(); }
});

/* ── selection geometry (shared by lasso + auto) ───────────────── */

function selectionPolygons() { // -> [[[x, y], ...], ...] in full-image px
  if (state.auto) {
    return state.auto.contours.map((c) => {
      const poly = [];
      for (let i = 0; i < c.length; i += 2) poly.push([c[i] * state.wsx, c[i + 1] * state.wsy]);
      return poly;
    });
  }
  const poly = [];
  for (const seg of state.segments) {
    for (let i = 0; i < seg.length; i += 2) {
      poly.push([(seg[i] + 0.5) * state.wsx, (seg[i + 1] + 0.5) * state.wsy]);
    }
  }
  return poly.length ? [poly] : [];
}

function invalidateSelection() { state.selPath = null; }

function selectionPath() { // cached Path2D pair: the selection, and "everything but" it
  if (state.selPath) return state.selPath;
  const sel = new Path2D();
  for (const poly of selectionPolygons()) {
    poly.forEach(([x, y], i) => (i === 0 ? sel.moveTo(x, y) : sel.lineTo(x, y)));
    sel.closePath();
  }
  const dim = new Path2D();
  dim.rect(0, 0, state.fullW, state.fullH);
  dim.addPath(sel);
  state.selPath = { sel, dim };
  return state.selPath;
}

/* ── rendering ─────────────────────────────────────────────────── */

let needsRender = true;
const requestRender = () => { needsRender = true; };
let antsPhase = 0;

function frame(t) {
  if (state.closed && !reducedMotion) { antsPhase = t / 40; needsRender = true; }
  if (needsRender) { needsRender = false; render(); }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function tracePoly(points) { // Int32Array of work coords -> current path
  for (let i = 0; i < points.length; i += 2) {
    const x = (points[i] + 0.5) * state.wsx;
    const y = (points[i + 1] + 0.5) * state.wsy;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
}

function traceSelection() { // all committed segments as ONE continuous subpath
  let started = false;
  for (const seg of state.segments) {
    for (let i = 0; i < seg.length; i += 2) {
      const x = (seg[i] + 0.5) * state.wsx;
      const y = (seg[i + 1] + 0.5) * state.wsy;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
  }
}

function render() {
  const dpr = devicePixelRatio || 1;
  const w = els.canvas.clientWidth, h = els.canvas.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.bitmap) return;

  const v = state.view;
  ctx.setTransform(dpr * v.scale, 0, 0, dpr * v.scale, dpr * v.tx, dpr * v.ty);
  ctx.imageSmoothingEnabled = v.scale < 3;
  ctx.drawImage(state.bitmap, 0, 0);

  const lw = (px) => px / v.scale;

  const autoSel = state.auto && state.auto.contours.length;
  if (state.closed && (state.segments.length || autoSel)) {
    // dim everything outside the selection, then marching ants on it
    const { sel, dim } = selectionPath();
    ctx.fillStyle = 'rgba(30,31,28,0.62)';
    ctx.fill(dim, 'evenodd');
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(30,31,28,0.9)';
    ctx.lineWidth = lw(3);
    ctx.stroke(sel);
    ctx.strokeStyle = '#f92672';
    ctx.lineWidth = lw(1.6);
    ctx.setLineDash([lw(6), lw(5)]);
    ctx.lineDashOffset = -lw(antsPhase % 11);
    ctx.stroke(sel);
    ctx.setLineDash([]);
  } else if (state.auto) {
    // auto-select preview: green tint on the detected object + its outline
    if (autoSel) {
      const { sel } = selectionPath();
      ctx.fillStyle = 'rgba(166,226,46,0.26)';
      ctx.fill(sel, 'evenodd');
      ctx.lineJoin = 'round';
      ctx.strokeStyle = 'rgba(30,31,28,0.85)';
      ctx.lineWidth = lw(2.6);
      ctx.stroke(sel);
      ctx.strokeStyle = '#a6e22e';
      ctx.lineWidth = lw(1.4);
      ctx.stroke(sel);
    }
  } else {
    // committed segments: pink over a soft halo (two strokes — canvas
    // shadows behave inconsistently under transforms across browsers)
    if (state.segments.length) {
      ctx.beginPath();
      traceSelection();
      ctx.lineJoin = 'round';
      ctx.strokeStyle = 'rgba(249,38,114,0.28)';
      ctx.lineWidth = lw(5);
      ctx.stroke();
      ctx.strokeStyle = '#f92672';
      ctx.lineWidth = lw(2);
      ctx.stroke();
    }
    // live (uncommitted) path: green
    if (state.livePath && state.livePath.length >= 4) {
      ctx.beginPath();
      tracePoly(state.livePath);
      ctx.strokeStyle = '#a6e22e';
      ctx.lineWidth = lw(1.6);
      ctx.lineJoin = 'round';
      ctx.stroke();
    }
  }

  // auto-select seeds: green = include, pink = exclude; strokes as polylines
  if (state.auto && !state.closed) drawSeeds(lw);

  // anchors
  state.anchors.forEach((a, i) => {
    const half = lw(i === 0 && state.anchors.length >= 2 && !state.closed ? 4.5 : 3);
    const cx = (a.x + 0.5) * state.wsx, cy = (a.y + 0.5) * state.wsy;
    ctx.fillStyle = i === 0 ? '#f92672' : '#f8f8f2';
    ctx.strokeStyle = '#1e1f1c';
    ctx.lineWidth = lw(1);
    ctx.fillRect(cx - half, cy - half, half * 2, half * 2);
    ctx.strokeRect(cx - half, cy - half, half * 2, half * 2);
  });
}

function drawSeeds(lw) {
  const groups = new Map();
  for (const s of state.auto.seeds) {
    if (!groups.has(s.group)) groups.set(s.group, []);
    groups.get(s.group).push(s);
  }
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const pts of groups.values()) {
    const colour = pts[0].sign > 0 ? '#a6e22e' : '#f92672';
    const cx = (p) => (p.x + 0.5) * state.wsx, cy = (p) => (p.y + 0.5) * state.wsy;
    if (pts.length > 1) {
      ctx.beginPath();
      pts.forEach((p, i) => (i === 0 ? ctx.moveTo(cx(p), cy(p)) : ctx.lineTo(cx(p), cy(p))));
      ctx.strokeStyle = 'rgba(30,31,28,0.8)';
      ctx.lineWidth = lw(5);
      ctx.stroke();
      ctx.strokeStyle = colour;
      ctx.lineWidth = lw(3);
      ctx.stroke();
    }
    const p = pts[0];
    ctx.beginPath();
    ctx.arc(cx(p), cy(p), lw(3.5), 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.strokeStyle = '#1e1f1c';
    ctx.lineWidth = lw(1);
    ctx.stroke();
  }
  ctx.lineCap = 'butt';
}

new ResizeObserver(() => {
  const dpr = devicePixelRatio || 1;
  els.canvas.width = Math.max(1, Math.round(els.canvas.clientWidth * dpr));
  els.canvas.height = Math.max(1, Math.round(els.canvas.clientHeight * dpr));
  requestRender();
}).observe(els.stage);

/* ── cutout generation ─────────────────────────────────────────── */

function chaikin(pts, iterations) {
  // corner-cutting smoothing for a *closed* polygon
  let p = pts;
  for (let it = 0; it < iterations; it++) {
    const out = [];
    for (let i = 0; i < p.length; i++) {
      const a = p[i], b = p[(i + 1) % p.length];
      out.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
      out.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    p = out;
  }
  return p;
}

function buildCutout() {
  const polys = selectionPolygons().filter((p) => p.length >= 3).map((p) => chaikin(p, 2));
  if (!polys.length) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const poly of polys) {
    for (const [x, y] of poly) {
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  minX = Math.max(0, Math.floor(minX) - 1);
  minY = Math.max(0, Math.floor(minY) - 1);
  maxX = Math.min(state.fullW, Math.ceil(maxX) + 1);
  maxY = Math.min(state.fullH, Math.ceil(maxY) + 1);
  const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);

  const oc = document.createElement('canvas');
  oc.width = w; oc.height = h;
  const octx = oc.getContext('2d');
  octx.translate(-minX, -minY);
  octx.beginPath();
  for (const poly of polys) {
    poly.forEach(([x, y], i) => (i === 0 ? octx.moveTo(x, y) : octx.lineTo(x, y)));
    octx.closePath();
  }
  octx.fillStyle = '#fff';
  octx.fill('evenodd');                          // antialiased mask (holes stay holes)
  octx.globalCompositeOperation = 'source-in';   // keep image only inside it
  octx.drawImage(state.bitmap, 0, 0);

  publishCutout(oc, { srcCanvas: oc, baseCanvas: oc, trimmed: false, smoothed: 0 });
  if (Number(els.smooth.value) > 0) requestSmooth(oc); // slider is a persistent setting
}

function publishCutout(canvas, opts) {
  canvas.toBlob((blob) => {
    if (!blob) return;
    if (state.cutout) URL.revokeObjectURL(state.cutout.url);
    state.cutout = {
      blob, url: URL.createObjectURL(blob),
      w: canvas.width, h: canvas.height,
      srcCanvas: opts.srcCanvas,    // original cut — trims re-derive from it
      baseCanvas: opts.baseCanvas,  // post-trim — smoothing re-derives from it
      trimmed: opts.trimmed,
      smoothed: opts.smoothed,
    };
    els.resultImg.src = state.cutout.url;
    els.resultImg.style.display = 'block';
    els.resultPlaceholder.style.display = 'none';
    els.resultMeta.innerHTML =
      `cutout <span class="num">${canvas.width}</span>×<span class="num">${canvas.height}</span>px · ` +
      `<span class="num">${(blob.size / 1024).toFixed(0)}</span> kb png` +
      (opts.trimmed ? ` · trimmed @ tol <span class="num">${els.tol.value}</span>` : '') +
      (opts.smoothed ? ` · smoothed @ <span class="num">${opts.smoothed}</span>` : '');
    setStatus('ready', opts.smoothed
      ? 'outline smoothed — the slider re-smooths non-destructively'
      : opts.trimmed
        ? 'background trimmed — adjust tolerance to re-trim from the original cut'
        : 'region cut — b trims background, s smooths the outline');
    updateUi();
  }, 'image/png');
}

/* ── auto-trim (space / panel button) ──────────────────────────── */

function requestTrim() {
  if (state.trimBusy || !state.engineReady) return;
  let src = state.cutout ? state.cutout.srcCanvas : null;
  if (!src) {
    if (!state.bitmap) {
      setStatus('ready', 'load an image first — trim needs something to work on');
      return;
    }
    // no cut yet: trim the whole imported image
    src = document.createElement('canvas');
    src.width = state.fullW;
    src.height = state.fullH;
    src.getContext('2d').drawImage(state.bitmap, 0, 0);
  }
  state.trimSrc = src;
  const data = src.getContext('2d').getImageData(0, 0, src.width, src.height).data;
  state.trimBusy = true;
  showBusy('trimming background…');
  worker.postMessage(
    { type: 'trim', rgba: data, width: src.width, height: src.height, tolerance: Number(els.tol.value) },
    [data.buffer]
  );
}

function applyTrimmed(m) {
  const c = document.createElement('canvas');
  c.width = m.width;
  c.height = m.height;
  c.getContext('2d').putImageData(new ImageData(m.rgba, m.width, m.height), 0, 0);
  publishCutout(c, { srcCanvas: state.trimSrc, baseCanvas: c, trimmed: true, smoothed: 0 });
  if (Number(els.smooth.value) > 0) requestSmooth(c); // keep the smooth setting applied
}

/* ── outline smoothing (s / slider) ────────────────────────────── */

function wholeImageCanvas() {
  if (!state.bitmap) return null;
  const c = document.createElement('canvas');
  c.width = state.fullW;
  c.height = state.fullH;
  c.getContext('2d').drawImage(state.bitmap, 0, 0);
  return c;
}

function requestSmooth(baseOverride) {
  if (state.trimBusy || state.smoothBusy || !state.engineReady) return;
  const base = baseOverride || (state.cutout ? state.cutout.baseCanvas : wholeImageCanvas());
  if (!base) {
    setStatus('ready', 'load an image or cut a region first — nothing to smooth');
    return;
  }
  const amount = Number(els.smooth.value);
  if (amount <= 0) {
    // un-smooth: republish the base as-is
    if (state.cutout) {
      publishCutout(base, {
        srcCanvas: state.cutout.srcCanvas, baseCanvas: base,
        trimmed: state.cutout.trimmed, smoothed: 0,
      });
    }
    return;
  }
  state.smoothBase = base;
  const data = base.getContext('2d').getImageData(0, 0, base.width, base.height).data;
  state.smoothBusy = true;
  showBusy('smoothing outline…');
  worker.postMessage(
    { type: 'smooth', rgba: data, width: base.width, height: base.height, amount },
    [data.buffer]
  );
}

function applySmoothed(m) {
  const c = document.createElement('canvas');
  c.width = m.width;
  c.height = m.height;
  c.getContext('2d').putImageData(new ImageData(m.rgba, m.width, m.height), 0, 0);
  publishCutout(c, {
    srcCanvas: state.cutout ? state.cutout.srcCanvas : state.smoothBase,
    baseCanvas: state.smoothBase,
    trimmed: state.cutout ? state.cutout.trimmed : false,
    smoothed: Number(els.smooth.value),
  });
}

els.trim.addEventListener('click', requestTrim);
els.tol.addEventListener('input', () => { els.tolVal.textContent = els.tol.value; });
els.tol.addEventListener('change', () => requestTrim());
els.smooth.addEventListener('input', () => { els.smoothVal.textContent = els.smooth.value; });
els.smooth.addEventListener('change', () => requestSmooth());

els.download.addEventListener('click', () => {
  if (!state.cutout) return;
  const a = document.createElement('a');
  a.href = state.cutout.url;
  a.download = `magic-clip-${state.cutout.w}x${state.cutout.h}.png`;
  a.click();
});

els.copy.addEventListener('click', async () => {
  if (!state.cutout) return;
  try {
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': state.cutout.blob })]);
    setStatus('ready', 'cutout copied to clipboard');
  } catch (err) {
    setStatus('error', `clipboard write failed: ${err.message || err}`);
  }
});

updateUi();
