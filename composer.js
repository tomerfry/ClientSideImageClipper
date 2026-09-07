/* Independent, local composition workspace. Layer sources are immutable
 * snapshots so new cuts, smoothing and source imports cannot alter a layout. */
'use strict';

(() => {
  const el = (id) => document.getElementById(id);
  const view = el('compose-view');
  const ctx = view.getContext('2d');
  const model = { width: 1200, height: 800, layers: [], selected: null };
  const history = [];
  let active = false, nextId = 1, drag = null;
  let transform = { scale: 1, x: 0, y: 0 };
  const selected = () => model.layers.find((layer) => layer.id === model.selected);
  const message = (text) => { el('compose-message').textContent = text; };
  const remember = () => {
    history.push({ ...model, layers: model.layers.map((layer) => ({ ...layer })) });
    if (history.length > 40) history.shift();
  };

  function switchTab(compose) {
    active = compose;
    drag = null;
    el('clip-panel').hidden = compose;
    el('clip-actions').hidden = compose;
    el('compose-panel').hidden = !compose;
    for (const [id, on] of [['tab-clip', !compose], ['tab-compose', compose]]) {
      el(id).setAttribute('aria-selected', String(on));
      el(id).tabIndex = on ? 0 : -1;
    }
    if (compose) render();
  }

  function sync() {
    const layer = selected();
    for (const action of ['front', 'back', 'duplicate', 'remove']) el(`compose-${action}`).disabled = !layer;
    el('compose-undo').disabled = !history.length;
    el('compose-export').disabled = !model.layers.length;
    el('compose-width').value = model.width;
    el('compose-height').value = model.height;
    el('compose-layers').replaceChildren();
    for (const item of [...model.layers].reverse()) {
      const button = document.createElement('button');
      button.textContent = item.name;
      button.setAttribute('aria-pressed', String(item.id === model.selected));
      button.addEventListener('click', () => { model.selected = item.id; sync(); view.focus(); });
      el('compose-layers').appendChild(button);
    }
    render();
  }

  function corners(layer) {
    return [[layer.x, layer.y], [layer.x + layer.w, layer.y],
      [layer.x + layer.w, layer.y + layer.h], [layer.x, layer.y + layer.h]];
  }

  function paintLayers(target) {
    target.imageSmoothingEnabled = true;
    target.imageSmoothingQuality = 'high';
    for (const layer of model.layers) target.drawImage(layer.source, layer.x, layer.y, layer.w, layer.h);
  }

  function render() {
    if (!active) return;
    const width = view.clientWidth, height = view.clientHeight;
    if (!width || !height) return;
    const dpr = window.devicePixelRatio || 1;
    view.width = Math.round(width * dpr);
    view.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#1e1f1c';
    ctx.fillRect(0, 0, width, height);
    const scale = Math.max(.001, Math.min((width - 40) / model.width, (height - 40) / model.height));
    transform = { scale, x: (width - model.width * scale) / 2, y: (height - model.height * scale) / 2 };
    ctx.translate(transform.x, transform.y);
    ctx.scale(scale, scale);
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, model.width, model.height); ctx.clip();
    const tile = 14 / scale;
    for (let y = 0, row = 0; y < model.height; y += tile, row++) {
      for (let x = 0, col = 0; x < model.width; x += tile, col++) {
        ctx.fillStyle = (row + col) % 2 ? '#35362f' : '#292a24';
        ctx.fillRect(x, y, tile, tile);
      }
    }
    paintLayers(ctx);
    if (!model.layers.length) {
      ctx.fillStyle = '#a59f85'; ctx.textAlign = 'center'; ctx.font = `${14 / scale}px monospace`;
      ctx.fillText('Add cutouts to build your canvas', model.width / 2, model.height / 2);
    }
    ctx.restore();
    const layer = selected();
    if (layer) {
      ctx.strokeStyle = '#66d9ef'; ctx.lineWidth = 1.5 / scale;
      ctx.strokeRect(layer.x, layer.y, layer.w, layer.h);
      for (const [x, y] of corners(layer)) {
        ctx.fillStyle = '#66d9ef'; ctx.fillRect(x - 4 / scale, y - 4 / scale, 8 / scale, 8 / scale);
      }
    }
  }

  function add(source, name = 'Cutout') {
    const copy = document.createElement('canvas');
    copy.width = source.width; copy.height = source.height;
    copy.getContext('2d').drawImage(source, 0, 0);
    remember();
    const factor = Math.min(1, model.width * .75 / copy.width, model.height * .75 / copy.height);
    const w = copy.width * factor, h = copy.height * factor;
    const id = nextId++;
    model.layers.push({ id, name: `${name} · ${id}`, source: copy, x: (model.width - w) / 2, y: (model.height - h) / 2, w, h });
    model.selected = id;
    switchTab(true); sync();
    message('Image added. Drag to position it.');
  }

  async function importImages(files) {
    for (const file of files) {
      if (!file.type.startsWith('image/')) continue;
      let bitmap;
      try {
        bitmap = await createImageBitmap(file);
        add(bitmap, file.name || 'Pasted image');
      } catch (error) { message(`Could not open image: ${error.message}`); }
      finally { bitmap?.close(); }
    }
  }

  function point(event) {
    const rect = view.getBoundingClientRect();
    return { x: (event.clientX - rect.left - transform.x) / transform.scale,
      y: (event.clientY - rect.top - transform.y) / transform.scale };
  }

  view.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    view.focus();
    const p = point(event);
    const current = selected();
    const corner = current ? corners(current).findIndex(([x, y]) => Math.hypot(p.x - x, p.y - y) < 12 / transform.scale) : -1;
    let hit = corner >= 0 ? current : null;
    if (!hit && p.x >= 0 && p.y >= 0 && p.x <= model.width && p.y <= model.height) {
      hit = [...model.layers].reverse().find((layer) => p.x >= layer.x && p.x <= layer.x + layer.w && p.y >= layer.y && p.y <= layer.y + layer.h);
    }
    model.selected = hit?.id ?? null;
    if (hit) {
      drag = { pointer: event.pointerId, p, original: { ...hit }, corner, saved: false };
      view.setPointerCapture(event.pointerId);
    }
    sync();
  });
  view.addEventListener('pointermove', (event) => {
    if (!drag || event.pointerId !== drag.pointer) return;
    const p = point(event), layer = selected(), original = drag.original;
    if (!layer) return;
    if (!drag.saved) {
      if (Math.hypot(p.x - drag.p.x, p.y - drag.p.y) * transform.scale < 2) return;
      remember(); drag.saved = true;
    }
    if (drag.corner < 0) {
      layer.x = Math.max(-layer.w + 1, Math.min(model.width - 1, original.x + p.x - drag.p.x));
      layer.y = Math.max(-layer.h + 1, Math.min(model.height - 1, original.y + p.y - drag.p.y));
    } else {
      const [ax, ay] = corners(original)[(drag.corner + 2) % 4];
      const [cx, cy] = corners(original)[drag.corner];
      const dx = cx - ax, dy = cy - ay;
      const factor = Math.max(1 / Math.min(original.w, original.h), Math.min(
        8192 / Math.max(original.w, original.h),
        ((p.x - ax) * dx + (p.y - ay) * dy) / (dx * dx + dy * dy)));
      layer.w = original.w * factor; layer.h = original.h * factor;
      layer.x = dx < 0 ? ax - layer.w : ax;
      layer.y = dy < 0 ? ay - layer.h : ay;
    }
    render();
  });
  function endDrag(event) {
    if (!drag || event.pointerId !== drag.pointer) return;
    drag = null;
    if (view.hasPointerCapture(event.pointerId)) view.releasePointerCapture(event.pointerId);
    sync();
  }
  for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) view.addEventListener(event, endDrag);

  function action(kind) {
    drag = null;
    if (kind === 'undo') {
      if (history.length) Object.assign(model, history.pop());
    } else {
      const layer = selected();
      if (!layer) return;
      remember();
      const index = model.layers.indexOf(layer);
      if (kind === 'duplicate') {
        const id = nextId++;
        model.layers.push({ ...layer, id, name: `${layer.name} copy`, x: Math.min(model.width - 1, layer.x + 20), y: Math.min(model.height - 1, layer.y + 20) });
        model.selected = id;
      } else {
        model.layers.splice(index, 1);
        if (kind === 'front') model.layers.push(layer);
        else if (kind === 'back') model.layers.unshift(layer);
        else model.selected = model.layers.at(-1)?.id ?? null;
      }
    }
    sync();
  }
  for (const kind of ['undo', 'front', 'back', 'duplicate', 'remove']) el(`compose-${kind}`).addEventListener('click', () => action(kind));
  window.addEventListener('keydown', (event) => {
    if (!active || event.target.closest('input, select, textarea, button, [contenteditable="true"]')) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault(); action('undo'); return;
    }
    if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); action('remove'); return; }
    const layer = selected();
    const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (layer && moves[event.key]) {
      event.preventDefault(); remember();
      const [dx, dy] = moves[event.key], step = event.shiftKey ? 10 : 1;
      layer.x = Math.max(1 - layer.w, Math.min(model.width - 1, layer.x + dx * step));
      layer.y = Math.max(1 - layer.h, Math.min(model.height - 1, layer.y + dy * step));
      sync();
    }
  });
  el('compose-size').addEventListener('submit', (event) => {
    event.preventDefault();
    const width = Number(el('compose-width').value), height = Number(el('compose-height').value);
    if (![width, height].every((n) => Number.isInteger(n) && n >= 1 && n <= 4096)) return;
    remember(); model.width = width; model.height = height;
    for (const layer of model.layers) {
      layer.x = Math.max(1 - layer.w, Math.min(width - 1, layer.x));
      layer.y = Math.max(1 - layer.h, Math.min(height - 1, layer.y));
    }
    sync(); message(`Canvas resized to ${width} × ${height}.`);
  });
  el('compose-import').addEventListener('click', () => el('compose-files').click());
  el('compose-files').addEventListener('change', (event) => {
    importImages([...event.target.files]); event.target.value = '';
  });
  window.addEventListener('dragover', (event) => { if (active) event.preventDefault(); });
  window.addEventListener('drop', (event) => {
    if (!active) return;
    event.preventDefault(); importImages([...event.dataTransfer.files]);
  });
  window.addEventListener('paste', (event) => {
    if (!active || event.target.closest('input, textarea, [contenteditable="true"]')) return;
    const files = [...(event.clipboardData?.files || [])].filter((file) => file.type.startsWith('image/'));
    if (files.length) { event.preventDefault(); importImages(files); }
  });
  el('compose-export').addEventListener('click', () => {
    const output = document.createElement('canvas');
    output.width = model.width; output.height = model.height;
    paintLayers(output.getContext('2d'));
    output.toBlob((blob) => {
      if (!blob) { message('Export failed. Try a smaller canvas.'); return; }
      const url = URL.createObjectURL(blob), link = document.createElement('a');
      link.href = url; link.download = `magic-canvas-${output.width}x${output.height}.png`;
      link.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
      message('Canvas exported as a transparent PNG.');
    }, 'image/png');
  });
  for (const [id, compose] of [['tab-clip', false], ['tab-compose', true]]) {
    el(id).addEventListener('click', () => switchTab(compose));
    el(id).addEventListener('keydown', (event) => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const next = event.key === 'Home' ? false : event.key === 'End' ? true : !compose;
        switchTab(next); el(next ? 'tab-compose' : 'tab-clip').focus();
      }
    });
  }
  new ResizeObserver(render).observe(el('compose-stage'));
  window.clipperCanvas = { add, get active() { return active; } };
  sync();
})();
