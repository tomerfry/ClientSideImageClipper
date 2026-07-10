/* magic-clipper UI thread.
 *
 * Owns: image loading (open/drop/paste), the pan/zoom view transform,
 * magnetic-lasso path state, canvas rendering, and cutout generation.
 * The livewire engine itself (numpy/scipy Dijkstra) lives in worker.js
 * behind four messages: init / setImage / seed / path.
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
  const hasPath = state.anchors.length > 0;
  els.undo.disabled = !hasPath;
  els.reset.disabled = !hasPath;
  els.cut.disabled = !(state.anchors.length >= 2 && !state.closed);
  els.fit.disabled = !state.bitmap;
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
  if (e.button !== 0 || !state.imageReady || state.closed) return;

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
  if (!state.imageReady || state.closed || state.anchors.length === 0) return;
  state.wantPath = eventToWork(e);
  pumpPath();
});

els.canvas.addEventListener('pointerup', (e) => {
  if (state.pan) {
    state.pan = null;
    els.canvas.classList.remove('pan-active');
    try { els.canvas.releasePointerCapture(e.pointerId); } catch {}
  }
});

els.canvas.addEventListener('dblclick', (e) => {
  e.preventDefault();
  requestClose();
});
els.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

function commitStraight(pt) {
  const a = state.anchors[state.anchors.length - 1];
  state.segments.push(Int32Array.from([a.x, a.y, pt.x, pt.y]));
  state.anchors.push(pt);
  state.livePath = null;
  sendSeed(pt);
  updateUi();
  requestRender();
}

function requestClose() {
  if (state.closed || state.anchors.length < 2 || state.pendingCommit) return;
  if (state.seedBusy) return; // last anchor's tree not ready yet
  const first = state.anchors[0];
  state.pendingCommit = { x: first.x, y: first.y, close: true };
  showBusy('closing path…');
  sendPath(first.x, first.y, 'close');
}

function undoAnchor() {
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
});

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

  if (state.closed && state.segments.length) {
    // dim everything outside the selection, then marching ants on it
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, state.fullW, state.fullH);
    traceSelection();
    ctx.fillStyle = 'rgba(30,31,28,0.62)';
    ctx.fill('evenodd');
    ctx.restore();

    ctx.beginPath();
    traceSelection();
    ctx.closePath();
    ctx.strokeStyle = 'rgba(30,31,28,0.9)';
    ctx.lineWidth = lw(3);
    ctx.stroke();
    ctx.strokeStyle = '#f92672';
    ctx.lineWidth = lw(1.6);
    ctx.setLineDash([lw(6), lw(5)]);
    ctx.lineDashOffset = -lw(antsPhase % 11);
    ctx.stroke();
    ctx.setLineDash([]);
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
  const poly = [];
  for (const seg of state.segments) {
    for (let i = 0; i < seg.length; i += 2) {
      poly.push([(seg[i] + 0.5) * state.wsx, (seg[i + 1] + 0.5) * state.wsy]);
    }
  }
  if (poly.length < 3) return;
  const smooth = chaikin(poly, 2);

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of smooth) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
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
  smooth.forEach(([x, y], i) => (i === 0 ? octx.moveTo(x, y) : octx.lineTo(x, y)));
  octx.closePath();
  octx.fillStyle = '#fff';
  octx.fill();                                   // antialiased mask
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
