/* Interaction tests without a browser: real handlers, lightweight DOM/canvas
 * doubles. Pixel rendering and layout still require a browser smoke test. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class Element {
  constructor() {
    this.listeners = {}; this.children = []; this.attributes = {};
    this.clientWidth = 840; this.clientHeight = 640; this.value = '';
    this.draws = []; this.width = 0; this.height = 0;
  }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  emit(type, event = {}) {
    for (const callback of this.listeners[type] || []) callback({
      target: this, preventDefault() {}, ...event,
    });
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  replaceChildren() { this.children = []; }
  appendChild(child) { this.children.push(child); }
  focus() {}
  closest() { return null; }
  click() { this.emit('click'); }
  getBoundingClientRect() { return { left: 0, top: 0 }; }
  setPointerCapture() {}
  hasPointerCapture() { return false; }
  getContext() {
    return new Proxy({}, { get: (_, key) => key === 'drawImage'
      ? (...args) => this.draws.push(args) : () => {} });
  }
  toBlob(callback) { callback({ canvas: this }); }
}
const elements = new Map();
const el = (id) => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
const window = new Element();
window.devicePixelRatio = 1;
let exported;
const document = { getElementById: el, createElement: () => new Element() };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../composer.js'), 'utf8'), {
  window, document, ResizeObserver: class { observe() {} },
  URL: { createObjectURL(blob) { exported = blob.canvas; return 'blob:test'; }, revokeObjectURL() {} },
  setTimeout() {},
});
const api = window.clipperCanvas;
const source = new Element(); source.width = 200; source.height = 100;
api.add(source);
assert.equal(api.active, true);
assert.equal(el('clip-panel').hidden, true);
assert.equal(el('compose-layers').children.length, 1);
el('compose-export').click();
assert.equal(exported.width, 1200);
assert.equal(exported.height, 800);
assert.deepEqual(exported.draws[0].slice(1), [500, 350, 200, 100]);
assert.notEqual(exported.draws[0][0], source, 'must snapshot source');
assert.equal(exported.draws[0][0].draws[0][0], source);

el('compose-duplicate').click();
assert.equal(el('compose-layers').children.length, 2);
el('compose-back').click();
el('compose-export').click();
assert.equal(exported.draws[0][1], 520, 'duplicate moved to back');
el('compose-undo').click();
el('compose-export').click();
assert.equal(exported.draws[1][1], 520, 'undo restores stacking');

window.emit('keydown', { key: 'ArrowRight', shiftKey: true });
el('compose-export').click();
assert.equal(exported.draws[1][1], 530, 'Shift arrow nudges 10px');
window.emit('keydown', { key: 'Delete' });
assert.equal(el('compose-layers').children.length, 1);
el('compose-undo').click();
assert.equal(el('compose-layers').children.length, 2);

// View fits 1200x800 into 840x640: scale 2/3, offset (20, 53 1/3).
const view = el('compose-view');
const pointer = (x, y) => ({ clientX: 20 + x * 2 / 3, clientY: 160 / 3 + y * 2 / 3, pointerId: 1, button: 0 });
view.emit('pointerdown', pointer(600, 400));
view.emit('pointermove', pointer(630, 430));
view.emit('pointerup', pointer(630, 430));
el('compose-export').click();
assert.equal(Math.round(exported.draws[1][1]), 560);
assert.equal(Math.round(exported.draws[1][2]), 400);
view.emit('pointerdown', pointer(760, 500));
view.emit('pointermove', pointer(860, 550));
view.emit('pointerup', pointer(860, 550));
el('compose-export').click();
assert.equal(Math.round(exported.draws[1][3]), 300);
assert.equal(Math.round(exported.draws[1][4]), 150, 'resize preserves aspect ratio');
el('compose-undo').click();
el('compose-export').click();
assert.equal(exported.draws[1][3], 200);

el('tab-clip').click();
assert.equal(api.active, false);
el('tab-compose').click();
assert.equal(el('compose-layers').children.length, 2, 'tabs preserve composition');
el('compose-width').value = '640'; el('compose-height').value = '480';
el('compose-size').emit('submit');
el('compose-export').click();
assert.equal(exported.width, 640); assert.equal(exported.height, 480);
assert.equal(exported.draws.length, 2, 'export draws only layers, not handles/checkerboard');
el('compose-undo').click();
el('compose-export').click();
assert.equal(exported.width, 1200);
console.log('Canvas interactions passed: add, snapshot, layers, undo, move, resize, tabs, dimensions, export');
