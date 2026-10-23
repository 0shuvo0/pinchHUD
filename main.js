(() => {
'use strict';

/* =====================================================================
   CONFIG — tweak behaviour here
   ===================================================================== */
const CONFIG = {
  // MediaPipe hand tracking. Runs fully in the browser (WebAssembly + GPU).
  // Self-host these three files if you want the app to work offline.
  VISION_JS: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs',
  WASM_URL:  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
  MODEL_URL: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',

  MIRROR: true,            // selfie view by default
  PINCH_START: 0.32,       // thumb-to-index distance / palm size that counts as a pinch
  PINCH_HOLD: 0.50,        // looser threshold while a box is being drawn (stops flicker)
  START_FRAMES: 3,         // consecutive two-hand pinch frames needed to start a box
  LOST_GRACE: 6,           // frames a pinch may drop out before the box is placed
  MIN_RECT_PX: 40,         // boxes smaller than this are discarded
  MAX_RECTS: 10,           // oldest box is removed beyond this
  PROCESS_MAX_PX: 420,     // pixel-processing resolution for per-pixel effects (lower = faster)
  SHOW_LABELS: true,       // effect name tag on each box
  SHOW_PINCH_POINTS: true  // rings that follow your pinching fingers
};

/* =====================================================================
   EFFECTS — set enabled:false to take an effect out of the random pool.
   Add your own: write an fx function below and add a line here.
   An fx function receives { c, src, w, h, t, scale, state }:
     c      2D context of a work canvas that already holds the video under the box
     src    that work canvas element
     w, h   work canvas size in pixels
     t      time in seconds
     scale  work pixels per screen pixel
     state  an object that persists for the life of that box
   ===================================================================== */
   const FILTERS = [
    { id: 'psy',     name: 'Psychedelic', enabled: true,               render: fxPsychedelic },
    { id: 'motion',  name: 'Motion glow', enabled: true,               render: fxMotion },
  { id: 'matrix',  name: 'Matrix',      enabled: true, maxPx: 1100, render: fxMatrix },
  { id: 'sketch',  name: 'Pen sketch',  enabled: true,               render: fxSketch },
  { id: 'neon',    name: 'Neon edges',  enabled: true,               render: fxNeon },
//   { id: 'thermal', name: 'Thermal',     enabled: true,               render: fxThermal },
//   { id: 'glitch',  name: 'Glitch',      enabled: true,               render: fxGlitch },
  { id: 'pixel',   name: 'Pixel mosaic',enabled: true, maxPx: 1100, render: fxPixel },
//   { id: 'half',    name: 'Halftone',    enabled: true, maxPx: 1100, render: fxHalftone },
//   { id: 'blood',   name: 'Blood',       enabled: true,               render: fxBlood },
//   { id: 'mirror',  name: 'Mirror tile', enabled: true,               render: fxMirror },
//   { id: 'echo',    name: 'Echo',        enabled: true,               render: fxEcho },
  { id: 'ascii',   name: 'ASCII',       enabled: true, maxPx: 1100, render: fxAscii },
  // Motion effects: these animate on their own, even when you hold still
//   { id: 'ripple',  name: 'Ripple',      enabled: true,               render: fxRipple },
//   { id: 'vortex',  name: 'Vortex',      enabled: true,               render: fxVortex },
//   { id: 'kaleido', name: 'Kaleidoscope',enabled: true,               render: fxKaleido },
//   { id: 'warp',    name: 'Time warp',   enabled: true, maxPx: 320,    render: fxTimeWarp },
//   { id: 'melt',    name: 'Melt',        enabled: true,               render: fxMelt }
];

/*FX-START*/
/* ---------- shared helpers ---------- */
const TAU = Math.PI * 2;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
function smoothstep(a, b, x) { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

function makeLUT(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const p = i / 255;
    let k = 0;
    while (k < stops.length - 2 && p > stops[k + 1][0]) k++;
    const [p0, c0] = stops[k], [p1, c1] = stops[k + 1];
    const f = clamp((p - p0) / (p1 - p0), 0, 1);
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * f;
  }
  return lut;
}
const LUT_THERMAL = makeLUT([
  [0,    [6, 0, 36]],
  [0.22, [58, 0, 140]],
  [0.42, [196, 0, 96]],
  [0.62, [255, 92, 0]],
  [0.82, [255, 212, 30]],
  [1,    [255, 255, 232]]
]);
const LUT_RAINBOW = (() => {
  const l = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const p = i / 255;
    l[i * 3]     = 127.5 + 127.5 * Math.cos(TAU * p);
    l[i * 3 + 1] = 127.5 + 127.5 * Math.cos(TAU * (p + 0.333));
    l[i * 3 + 2] = 127.5 + 127.5 * Math.cos(TAU * (p + 0.667));
  }
  return l;
})();

// Small cached canvas stored on a box's state object.
function scratch(state, key, w, h) {
  let s = state[key];
  if (!s) {
    const cv = document.createElement('canvas');
    s = state[key] = { cv, cx: cv.getContext('2d', { willReadFrequently: true }), fresh: true };
  }
  if (s.cv.width !== w || s.cv.height !== h) { s.cv.width = w; s.cv.height = h; s.fresh = true; }
  else s.fresh = false;
  s.cx.imageSmoothingEnabled = true;
  s.cx.imageSmoothingQuality = 'high';
  return s;
}
function buf(state, key, Type, n) {
  if (!state[key] || state[key].length !== n) state[key] = new Type(n);
  return state[key];
}
function grayOf(img, state) {
  const n = img.width * img.height, d = img.data;
  const g = buf(state, 'g', Float32Array, n);
  for (let i = 0, p = 0; p < n; i += 4, p++) g[p] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  return g;
}
function sobel(g, w, h, state) {
  const m = buf(state, 'm', Float32Array, w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = -g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1] + g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1];
      const gy = -g[i - w - 1] - 2 * g[i - w] - g[i - w + 1] + g[i + w - 1] + 2 * g[i + w] + g[i + w + 1];
      m[i] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return m;
}


/* ---------- Matrix: falling glyphs lit by the video's brightness ---------- */
const MATRIX_CHARS = 'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾅﾆﾇﾈﾊﾋﾓﾔﾕﾗﾘﾜ0123456789Z:.=*+-<>'.split('');
function matrixResetCol(m, i, rows, initial) {
  m.head[i] = initial ? -Math.random() * rows * 0.6 + Math.random() * rows : -Math.random() * rows * 0.5;
  m.speed[i] = 7 + Math.random() * 17;
  m.trail[i] = 6 + Math.random() * 14;
}
function fxMatrix(o) {
  const { c, src, w, h, t, scale, state } = o;
  const cell = Math.max(6, Math.round(11 * scale));       // smaller glyphs = more picture detail
  const cols = Math.max(1, Math.ceil(w / cell)), rows = Math.max(1, Math.ceil(h / cell));
  let m = state.m;
  if (!m || m.cols !== cols || m.rows !== rows) {
    const old = m;
    m = state.m = {
      cols, rows, last: t,
      lo: old ? old.lo : 0, hi: old ? old.hi : 255,
      head: new Float32Array(cols), speed: new Float32Array(cols), trail: new Float32Array(cols),
      ch: new Uint8Array(cols * rows)
    };
    for (let i = 0; i < cols; i++) {
      if (old && i < old.cols) { m.head[i] = old.head[i]; m.speed[i] = old.speed[i]; m.trail[i] = old.trail[i]; }
      else matrixResetCol(m, i, rows, true);
    }
    for (let i = 0; i < m.ch.length; i++) m.ch[i] = (Math.random() * MATRIX_CHARS.length) | 0;
  }
  const dt = clamp(t - m.last, 0, 0.1);
  m.last = t;
  for (let i = 0; i < cols; i++) {
    m.head[i] += m.speed[i] * dt;
    if (m.head[i] - m.trail[i] > rows) matrixResetCol(m, i, rows, false);
  }
 
  // Sample the video: one pixel per glyph cell.
  const tiny = scratch(state, 'tiny', cols, rows);
  tiny.cx.drawImage(src, 0, 0, w, h, 0, 0, cols, rows);
  const img = tiny.cx.getImageData(0, 0, cols, rows), px = img.data;
  const n = cols * rows;
  const lum = buf(state, 'lum', Float32Array, n);
 
  // Auto-levels: stretch the 4th..96th brightness percentiles to 0..1 so a face always has contrast.
  const hist = new Uint16Array(64);
  for (let p = 0; p < n; p++) {
    const l = 0.299 * px[p * 4] + 0.587 * px[p * 4 + 1] + 0.114 * px[p * 4 + 2];
    lum[p] = l;
    hist[Math.min(63, (l / 4) | 0)]++;
  }
  let acc = 0, lo = 0, hi = 255;
  for (let k = 0; k < 64; k++) { acc += hist[k]; if (acc >= n * 0.04) { lo = k * 4; break; } }
  acc = 0;
  for (let k = 63; k >= 0; k--) { acc += hist[k]; if (acc >= n * 0.04) { hi = (k + 1) * 4; break; } }
  if (hi - lo < 40) hi = lo + 40;
  m.lo += (lo - m.lo) * 0.12;
  m.hi += (hi - m.hi) * 0.12;
  const span = m.hi - m.lo;
  for (let p = 0; p < n; p++) {
    const v = clamp((lum[p] - m.lo) / span, 0, 1);
    lum[p] = v * v * (3 - 2 * v);                           // gentle S-curve for punchier shapes
  }
 
  // Underlay: a soft green-tinted copy of the video, so faces and shapes stay readable between glyphs.
  for (let p = 0; p < n; p++) {
    const v = lum[p];
    px[p * 4] = v * 12; px[p * 4 + 1] = v * 125; px[p * 4 + 2] = v * 38; px[p * 4 + 3] = 255;
  }
  tiny.cx.putImageData(img, 0, 0);
  c.fillStyle = '#010402';
  c.fillRect(0, 0, w, h);
  c.imageSmoothingEnabled = true;
  c.drawImage(tiny.cv, 0, 0, cols, rows, 0, 0, cols * cell, rows * cell);
 
  // Falling glyphs: brightness comes from the video, the rain adds a lift on top.
  c.font = cell + 'px "Courier New", monospace';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  for (let x = 0; x < cols; x++) {
    const head = m.head[x], tr = m.trail[x];
    for (let y = 0; y < rows; y++) {
      const i = y * cols + x;
      if (Math.random() < 0.02) m.ch[i] = (Math.random() * MATRIX_CHARS.length) | 0;
      const d = head - y;
      let b = 0;
      if (d >= 0 && d < tr) { b = 1 - d / tr; b *= b; }
      const v = lum[i];
      const isHead = d >= 0 && d < 1.2;
      let a = v * (0.42 + 0.95 * b) + b * 0.16;
      if (isHead) a = 0.6 + 0.4 * v;
      if (a < 0.07) continue;
      if (a > 1) a = 1;
      c.fillStyle = isHead
        ? 'rgba(220,255,230,' + a.toFixed(2) + ')'
        : 'rgba(' + ((v * 90) | 0) + ',255,' + (70 + ((v * 110) | 0)) + ',' + a.toFixed(2) + ')';
      c.fillText(MATRIX_CHARS[m.ch[i]], x * cell + cell / 2, y * cell + cell / 2);
    }
  }
}
 
/* ---------- Pen sketch: dark ink edges on warm paper ---------- */
function fxSketch(o) {
  const { c, w, h, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const m = sobel(grayOf(img, state), w, h, state);
  for (let p = 0, i = 0; p < w * h; p++, i += 4) {
    const v = 255 - clamp((m[p] - 22) * 1.7, 0, 255);
    d[i] = v; d[i + 1] = v * 0.985; d[i + 2] = v * 0.945; d[i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Neon edges: glowing rainbow outlines ---------- */
function fxNeon(o) {
  const { c, w, h, t, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const m = sobel(grayOf(img, state), w, h, state);
  const drift = t * 0.18;
  for (let y = 0, p = 0; y < h; y++) {
    for (let x = 0; x < w; x++, p++) {
      const i = p * 4;
      const e = clamp((m[p] - 16) / 150, 0, 1);
      let ph = (x / w) * 0.6 + (y / h) * 0.4 + drift;
      ph -= Math.floor(ph);
      const k = ((ph * 255) | 0) * 3;
      d[i]     = LUT_RAINBOW[k]     * e * 1.15 + d[i]     * 0.09;
      d[i + 1] = LUT_RAINBOW[k + 1] * e * 1.15 + d[i + 1] * 0.09;
      d[i + 2] = LUT_RAINBOW[k + 2] * e * 1.15 + d[i + 2] * 0.09;
      d[i + 3] = 255;
    }
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Psychedelic: brightness mapped onto a drifting rainbow ---------- */
function fxPsychedelic(o) {
  const { c, w, h, t, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const wave = buf(state, 'wave', Float32Array, w);
  for (let x = 0; x < w; x++) wave[x] = Math.sin(x * 0.03 + t) * 0.04;
  for (let y = 0, p = 0; y < h; y++) {
    const rowShift = (y / h) * 0.12 + t * 0.1;
    for (let x = 0; x < w; x++, p++) {
      const i = p * 4;
      const l = smoothstep(0.05, 0.95, (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255);
      let ph = l * 1.5 + rowShift + wave[x];
      ph -= Math.floor(ph);
      const k = ((ph * 255) | 0) * 3;
      d[i]     = LUT_RAINBOW[k]     * 0.82 + d[i]     * 0.18;
      d[i + 1] = LUT_RAINBOW[k + 1] * 0.82 + d[i + 1] * 0.18;
      d[i + 2] = LUT_RAINBOW[k + 2] * 0.82 + d[i + 2] * 0.18;
    }
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Thermal: heat-camera palette ---------- */
function fxThermal(o) {
  const { c, w, h } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const l = clamp(((0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255 - 0.5) * 1.25 + 0.5, 0, 1);
    const k = ((l * 255) | 0) * 3;
    d[i] = LUT_THERMAL[k]; d[i + 1] = LUT_THERMAL[k + 1]; d[i + 2] = LUT_THERMAL[k + 2];
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Glitch: RGB split, torn rows, scanlines ---------- */
function fxGlitch(o) {
  const { c, w, h, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const src = buf(state, 'copy', Uint8ClampedArray, w * h * 4);
  src.set(d);
  const rows = buf(state, 'rowOff', Int16Array, h);
  state.n = (state.n || 0) + 1;
  if (state.n % 4 === 1 || state.rowsH !== h) {
    state.rowsH = h;
    rows.fill(0);
    const bands = 1 + ((Math.random() * 4) | 0);
    for (let k = 0; k < bands; k++) {
      const y0 = (Math.random() * h) | 0;
      const bh = (2 + Math.random() * h * 0.14) | 0;
      const off = ((Math.random() - 0.5) * w * 0.35) | 0;
      for (let y = y0; y < Math.min(h, y0 + bh); y++) rows[y] = off;
    }
    state.shift = Math.round(w * (0.008 + Math.random() * 0.025));
  }
  const sh = state.shift;
  for (let y = 0; y < h; y++) {
    const off = rows[y], scan = (y % 3 === 0) ? 0.82 : 1;
    for (let x = 0; x < w; x++) {
      const xr = clamp(x + off + sh, 0, w - 1), xg = clamp(x + off, 0, w - 1), xb = clamp(x + off - sh, 0, w - 1);
      const i = (y * w + x) * 4;
      d[i]     = src[(y * w + xr) * 4]     * scan;
      d[i + 1] = src[(y * w + xg) * 4 + 1] * scan;
      d[i + 2] = src[(y * w + xb) * 4 + 2] * scan;
      d[i + 3] = 255;
    }
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Pixel mosaic: chunky, colour-reduced blocks ---------- */
function fxPixel(o) {
  const { c, src, w, h, scale, state } = o;
  const block = Math.max(4, Math.round(13 * scale));
  const tw = Math.max(1, Math.ceil(w / block)), th = Math.max(1, Math.ceil(h / block));
  const tiny = scratch(state, 'tiny', tw, th);
  tiny.cx.drawImage(src, 0, 0, w, h, 0, 0, tw, th);
  const img = tiny.cx.getImageData(0, 0, tw, th), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    for (let j = 0; j < 3; j++) {
      const v = g + (d[i + j] - g) * 1.35;
      d[i + j] = Math.round(clamp(v, 0, 255) / 51) * 51;
    }
  }
  tiny.cx.putImageData(img, 0, 0);
  c.imageSmoothingEnabled = false;
  c.drawImage(tiny.cv, 0, 0, tw, th, 0, 0, tw * block, th * block);
}

/* ---------- Halftone: coloured dots sized by brightness ---------- */
function fxHalftone(o) {
  const { c, src, w, h, scale, state } = o;
  const cell = Math.max(5, Math.round(9 * scale));
  const cols = Math.max(1, Math.ceil(w / cell)), rows = Math.max(1, Math.ceil(h / cell));
  const tiny = scratch(state, 'tiny', cols, rows);
  tiny.cx.drawImage(src, 0, 0, w, h, 0, 0, cols, rows);
  const px = tiny.cx.getImageData(0, 0, cols, rows).data;
  c.fillStyle = '#0c0d12';
  c.fillRect(0, 0, w, h);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = (y * cols + x) * 4;
      const r = px[i], g = px[i + 1], b = px[i + 2];
      const l = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      const rad = cell * 0.66 * Math.sqrt(clamp(l * 1.15, 0, 1));
      if (rad < 0.7) continue;
      c.fillStyle = 'rgb(' + Math.min(255, r * 1.25) + ',' + Math.min(255, g * 1.25) + ',' + Math.min(255, b * 1.25) + ')';
      c.beginPath();
      c.arc(x * cell + cell / 2, y * cell + cell / 2, rad, 0, TAU);
      c.fill();
    }
  }
}

/* ---------- Blood: dark areas run red, light areas wash out white ---------- */
function fxBlood(o) {
  const { c, w, h } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const l = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
    const s = smoothstep(0.28, 0.62, l);
    const k = Math.pow(s, 1.6) * 255;
    d[i] = 165 + 90 * s; d[i + 1] = k; d[i + 2] = k * 0.97; d[i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Mirror tile: the box folds into four reflected copies ---------- */
function fxMirror(o) {
  const { c, src, w, h, state } = o;
  const tmp = scratch(state, 'tmp', w, h);
  tmp.cx.drawImage(src, 0, 0);
  const hw = Math.ceil(w / 2), hh = Math.ceil(h / 2);
  c.clearRect(0, 0, w, h);
  c.drawImage(tmp.cv, 0, 0, w, h, 0, 0, hw, hh);
  c.save(); c.translate(w, 0); c.scale(-1, 1); c.drawImage(tmp.cv, 0, 0, w, h, 0, 0, hw, hh); c.restore();
  c.save(); c.translate(0, h); c.scale(1, -1); c.drawImage(tmp.cv, 0, 0, w, h, 0, 0, hw, hh); c.restore();
  c.save(); c.translate(w, h); c.scale(-1, -1); c.drawImage(tmp.cv, 0, 0, w, h, 0, 0, hw, hh); c.restore();
}

/* ---------- Echo: motion leaves ghost trails ---------- */
function fxEcho(o) {
  const { c, src, w, h, state } = o;
  const acc = scratch(state, 'acc', w, h);
  if (acc.fresh) {
    acc.cx.globalAlpha = 1;
    acc.cx.drawImage(src, 0, 0);
  } else {
    acc.cx.globalAlpha = 0.13;
    acc.cx.drawImage(src, 0, 0);
    acc.cx.globalAlpha = 1;
  }
  c.globalAlpha = 0.72;
  c.drawImage(acc.cv, 0, 0);
  c.globalAlpha = 1;
}

/* ---------- ASCII: the video drawn in text ---------- */
const ASCII_RAMP = ' .:-=+*#%@';
function fxAscii(o) {
  const { c, src, w, h, scale, state } = o;
  const cw = Math.max(5, Math.round(8 * scale)), ch = Math.max(8, Math.round(cw * 1.8));
  const cols = Math.max(1, Math.floor(w / cw)), rows = Math.max(1, Math.floor(h / ch));
  const tiny = scratch(state, 'tiny', cols, rows);
  tiny.cx.drawImage(src, 0, 0, w, h, 0, 0, cols, rows);
  const px = tiny.cx.getImageData(0, 0, cols, rows).data;
  c.fillStyle = '#050607';
  c.fillRect(0, 0, w, h);
  c.font = Math.round(ch * 0.95) + 'px "Courier New", monospace';
  c.textAlign = 'left';
  c.textBaseline = 'top';
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = (y * cols + x) * 4;
      const r = px[i], g = px[i + 1], b = px[i + 2];
      const l = clamp(((0.299 * r + 0.587 * g + 0.114 * b) / 255 - 0.08) * 1.35, 0, 1);
      const idx = Math.min(ASCII_RAMP.length - 1, (l * ASCII_RAMP.length) | 0);
      if (idx === 0) continue;
      c.fillStyle = 'rgb(' + Math.min(255, r * 1.4) + ',' + Math.min(255, g * 1.4) + ',' + Math.min(255, b * 1.4) + ')';
      c.fillText(ASCII_RAMP[idx], x * cw, y * ch);
    }
  }
}
/* =====================================================================
   MOTION EFFECTS — these keep moving even when you hold still
   ===================================================================== */
// Smooth pixel lookup, writes into d at index i.
function bilin(src, w, h, x, y, d, i) {
  x = clamp(x, 0, w - 1.001); y = clamp(y, 0, h - 1.001);
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0;
  const i00 = (y0 * w + x0) * 4, i10 = i00 + 4, i01 = i00 + w * 4, i11 = i01 + 4;
  const a = (1 - fx) * (1 - fy), b = fx * (1 - fy), e = (1 - fx) * fy, f = fx * fy;
  d[i]     = src[i00]     * a + src[i10]     * b + src[i01]     * e + src[i11]     * f;
  d[i + 1] = src[i00 + 1] * a + src[i10 + 1] * b + src[i01 + 1] * e + src[i11 + 1] * f;
  d[i + 2] = src[i00 + 2] * a + src[i10 + 2] * b + src[i01 + 2] * e + src[i11 + 2] * f;
  d[i + 3] = 255;
}

/* ---------- Ripple: waves roll out from a wandering centre ---------- */
function fxRipple(o) {
  const { c, w, h, t, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const src = buf(state, 'copy', Uint8ClampedArray, w * h * 4);
  src.set(d);
  const m = Math.min(w, h), big = Math.max(w, h);
  const amp = m * 0.045, k = TAU / (m * 0.2), speed = t * 5;
  const cx = w / 2 + Math.sin(t * 0.8) * w * 0.22, cy = h / 2 + Math.cos(t * 0.6) * h * 0.22;
  for (let y = 0, p = 0; y < h; y++) {
    const sway = Math.sin(y * 0.05 + t * 3) * amp * 0.3;
    for (let x = 0; x < w; x++, p++) {
      const dx = x - cx, dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy) + 0.0001;
      const wv = Math.sin(r * k - speed) * amp * Math.exp(-r / (big * 0.9));
      bilin(src, w, h, x + (dx / r) * wv + sway, y + (dy / r) * wv, d, p * 4);
    }
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Vortex: a twirl that winds and unwinds ---------- */
function fxVortex(o) {
  const { c, w, h, t, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const src = buf(state, 'copy', Uint8ClampedArray, w * h * 4);
  src.set(d);
  const R = Math.min(w, h) * 0.64;
  const cx = w / 2 + Math.sin(t * 0.7) * w * 0.16, cy = h / 2 + Math.cos(t * 0.9) * h * 0.16;
  const strength = Math.sin(t * 1.1) * 3.4 + Math.sin(t * 0.37) * 1.2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx, dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy);
      if (r >= R) continue;                      // outside the swirl the pixel is already correct
      const f = 1 - r / R, a = strength * f * f;
      const ca = Math.cos(a), sa = Math.sin(a);
      bilin(src, w, h, cx + dx * ca - dy * sa, cy + dx * sa + dy * ca, d, (y * w + x) * 4);
    }
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Kaleidoscope: spinning, breathing mirror wedges ---------- */
function fxKaleido(o) {
  const { c, w, h, t, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const src = buf(state, 'copy', Uint8ClampedArray, w * h * 4);
  src.set(d);
  let P = state.polar;
  if (!P || P.w !== w || P.h !== h) {
    P = state.polar = { w, h, ang: new Float32Array(w * h), rad: new Float32Array(w * h) };
    for (let y = 0, p = 0; y < h; y++) {
      for (let x = 0; x < w; x++, p++) {
        const dx = x - w / 2, dy = y - h / 2;
        P.ang[p] = Math.atan2(dy, dx);
        P.rad[p] = Math.sqrt(dx * dx + dy * dy);
      }
    }
  }
  const seg = TAU / 6, half = seg / 2;
  const rot = t * 0.45, off = Math.sin(t * 0.6) * 0.7, zoom = 0.6 + 0.16 * Math.sin(t * 0.8);
  const cx = w / 2 + Math.sin(t * 0.5) * w * 0.08, cy = h / 2 + Math.cos(t * 0.45) * h * 0.08;
  for (let p = 0; p < w * h; p++) {
    let a = P.ang[p] + rot;
    a -= Math.floor(a / seg) * seg;
    if (a > half) a = seg - a;
    a += off;
    const r = P.rad[p] * zoom;
    bilin(src, w, h, cx + Math.cos(a) * r, cy + Math.sin(a) * r, d, p * 4);
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Motion glow: only what moves lights up, and leaves a rainbow wake ---------- */
function fxMotion(o) {
  const { c, w, h, t, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const n = w * h;
  const prev = buf(state, 'gprev', Float32Array, n);
  const tr = buf(state, 'trail', Float32Array, n);
  const fresh = state.mn !== n;
  state.mn = n;
  const hue = t * 0.15;
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    if (fresh) prev[p] = g;
    const diff = Math.abs(g - prev[p]);
    prev[p] += (g - prev[p]) * 0.5;
    const amp = clamp((diff - 9) / 55, 0, 1);
    const v = Math.max(tr[p] * 0.9, amp);
    tr[p] = v;
    let ph = hue + (1 - v) * 0.4;
    ph -= Math.floor(ph);
    const k = ((ph * 255) | 0) * 3;
    d[i]     = d[i]     * 0.26 + LUT_RAINBOW[k]     * v * 1.4;
    d[i + 1] = d[i + 1] * 0.26 + LUT_RAINBOW[k + 1] * v * 1.4;
    d[i + 2] = d[i + 2] * 0.26 + LUT_RAINBOW[k + 2] * v * 1.4;
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Time warp: each row shows a different moment, and the bands sweep ---------- */
function fxTimeWarp(o) {
  const { c, w, h, t, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const N = 32;
  let F = state.tw;
  if (!F || F.w !== w || F.h !== h) F = state.tw = { w, h, i: 0, n: 0, frames: [] };
  F.i = (F.i + 1) % N;
  if (!F.frames[F.i]) F.frames[F.i] = new Uint8ClampedArray(w * h * 4);
  F.frames[F.i].set(d);
  F.n = Math.min(N, F.n + 1);
  const rowBytes = w * 4;
  for (let y = 0; y < h; y++) {
    const phase = 0.5 + 0.5 * Math.sin((y / h) * TAU * 0.9 + t * 1.3);
    const delay = Math.floor(phase * (F.n - 1));
    const fr = F.frames[(F.i - delay + N) % N];
    d.set(fr.subarray(y * rowBytes, (y + 1) * rowBytes), y * rowBytes);
  }
  c.putImageData(img, 0, 0);
}

/* ---------- Melt: the picture drips downward in waves, then pulls back ---------- */
function fxMelt(o) {
  const { c, w, h, t, state } = o;
  const img = c.getImageData(0, 0, w, h), d = img.data;
  const src = buf(state, 'copy', Uint8ClampedArray, w * h * 4);
  src.set(d);
  const amount = 0.55 + 0.45 * Math.sin(t * 0.9);
  const col = buf(state, 'col', Float32Array, w);
  for (let x = 0; x < w; x++) {
    const n = (Math.sin(x * 0.09 + t * 1.3) + 0.6 * Math.sin(x * 0.23 - t * 0.9) + 0.3 * Math.sin(x * 0.51 + t * 2.1) + 1.9) / 3.8;
    col[x] = n * amount * h * 0.5;
  }
  for (let y = 0; y < h; y++) {
    const k = y / h;
    const wobble = Math.sin(y * 0.08 + t * 2) * 1.5 * k;
    for (let x = 0; x < w; x++) bilin(src, w, h, x + wobble, y - col[x] * k, d, (y * w + x) * 4);
  }
  c.putImageData(img, 0, 0);
}
/*FX-END*/

/* =====================================================================
   APP
   ===================================================================== */
const $ = (s) => document.querySelector(s);
const video = $('#cam'), canvas = $('#stage'), ctx = canvas.getContext('2d');
const work = document.createElement('canvas');
const wctx = work.getContext('2d', { willReadFrequently: true });

const el = {
  welcome: $('#welcome'), setup: $('#setup'), loader: $('#loader'), hud: $('#hud'),
  btnCam: $('#btnCam'), camError: $('#camError'),
  zoom: $('#zoom'), zoomVal: $('#zoomVal'), mirror: $('#mirror'), btnReset: $('#btnReset'), btnStart: $('#btnStart'),
  modelChip: $('#modelChip'), modelText: $('#modelText'),
  loaderTitle: $('#loaderTitle'), loaderSub: $('#loaderSub'),
  pill: $('#statusPill'), statusText: $('#statusText'), hint: $('#hint'), toast: $('#toast')
};

const app = {
  mode: 'welcome',          // welcome | setup | live
  stream: null,
  model: null, modelState: 'idle', modelPromise: null,
  rects: [], activeRect: null,
  hands: [], pinchCount: 0,
  showSkeleton: false,
  lastVideoTime: -1
};
const crop = { zoom: 1, cx: 0.5, cy: 0.5, mirror: CONFIG.MIRROR };
const G = { phase: 'idle', on: 0, lost: 0, rel: 0 };   // gesture state machine: idle | drawing | release
let W = 0, H = 0, DPR = 1;

/* ---------- layout ---------- */
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = Math.round(W * DPR);
  canvas.height = Math.round(H * DPR);
}
window.addEventListener('resize', resize);
resize();

/* ---------- toast / status ---------- */
let toastTimer = 0;
function toast(msg, ms = 4200) {
  el.toast.textContent = msg;
  el.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove('show'), ms);
}
let lastStatus = '';
function setStatus(text, state) {
  const key = state + '|' + text;
  if (key === lastStatus) return;
  lastStatus = key;
  el.statusText.textContent = text;
  el.pill.dataset.state = state;
}

/* ---------- camera ---------- */
async function openCamera() {
  el.camError.hidden = true;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return camFail('The camera needs a secure page. Open this file from localhost or an https address.');
  }
  el.btnCam.disabled = true;
  try {
    app.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    video.srcObject = app.stream;
    await video.play();
  } catch (e) {
    el.btnCam.disabled = false;
    if (e && e.name === 'NotAllowedError') return camFail('Camera access was blocked. Allow it in your browser’s site settings, then try again.');
    if (e && e.name === 'NotFoundError') return camFail('No camera was found on this device.');
    return camFail('The camera could not start: ' + ((e && e.message) || 'unknown error') + '.');
  }
  el.welcome.hidden = true;
  el.setup.hidden = false;
  app.mode = 'setup';
  app.modelPromise = loadModel();     // start downloading the model while the person frames the shot
}
function camFail(msg) {
  el.camError.textContent = msg;
  el.camError.hidden = false;
}

/* ---------- hand tracking model ---------- */
function setModelUI(state, text) {
  el.modelChip.dataset.state = state;
  el.modelText.textContent = text;
  el.modelChip.disabled = state !== 'failed';
}
async function loadModel() {
  if (app.modelState === 'loading') return app.modelPromise;
  app.modelState = 'loading';
  setModelUI('loading', 'Loading hand tracking');
  el.loaderTitle.textContent = 'Loading hand tracking';
  try {
    const mod = await import(CONFIG.VISION_JS);
    el.loaderTitle.textContent = 'Starting the hand tracker';
    const fileset = await mod.FilesetResolver.forVisionTasks(CONFIG.WASM_URL);
    const make = (delegate) => mod.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: CONFIG.MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    try { app.model = await make('GPU'); }
    catch (gpuErr) { console.warn('GPU delegate unavailable, using CPU.', gpuErr); app.model = await make('CPU'); }
    app.modelState = 'ready';
    setModelUI('ready', 'Hand tracking ready');
  } catch (err) {
    console.error('Hand tracking failed to load', err);
    app.modelState = 'failed';
    setModelUI('failed', 'Failed to load. Retry');
  }
}
el.modelChip.addEventListener('click', () => {
  if (app.modelState === 'failed') { app.modelState = 'idle'; app.modelPromise = loadModel(); }
});

/* ---------- crop and zoom view ---------- */
function getView() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const va = vw / vh, wa = W / H;
  let bw, bh;
  if (va > wa) { bh = vh; bw = vh * wa; } else { bw = vw; bh = vw / wa; }
  const sw = bw / crop.zoom, sh = bh / crop.zoom;
  const sx = clamp(crop.cx * vw - sw / 2, 0, vw - sw);
  const sy = clamp(crop.cy * vh - sh / 2, 0, vh - sh);
  crop.cx = (sx + sw / 2) / vw;
  crop.cy = (sy + sh / 2) / vh;
  return { W, H, vw, vh, sx, sy, sw, sh, mirror: crop.mirror };
}
// Draw the part of the view covering screen rect (x,y,w,h) into context c, scaled to dw x dh.
function drawVideoRegion(c, v, x, y, w, h, dw, dh) {
  const sx = v.sx + ((v.mirror ? (v.W - x - w) : x) / v.W) * v.sw;
  const sy = v.sy + (y / v.H) * v.sh;
  const sw = (w / v.W) * v.sw, sh = (h / v.H) * v.sh;
  c.save();
  if (v.mirror) { c.translate(dw, 0); c.scale(-1, 1); }
  c.drawImage(video, sx, sy, sw, sh, 0, 0, dw, dh);
  c.restore();
}
function toCanvas(lx, ly, v) {
  let x = ((lx * v.vw - v.sx) / v.sw) * v.W;
  if (v.mirror) x = v.W - x;
  const y = ((ly * v.vh - v.sy) / v.sh) * v.H;
  return { x, y };
}

/* ---------- boxes ---------- */
let bag = [], lastFilterId = null;
function nextFilter() {
  const enabled = FILTERS.filter((f) => f.enabled);
  if (!enabled.length) { toast('Every effect is switched off in FILTERS.'); return null; }
  bag = bag.filter((f) => f.enabled);
  if (!bag.length) {
    bag = enabled.slice();
    for (let i = bag.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      [bag[i], bag[j]] = [bag[j], bag[i]];
    }
    if (bag.length > 1 && bag[bag.length - 1].id === lastFilterId) [bag[0], bag[bag.length - 1]] = [bag[bag.length - 1], bag[0]];
  }
  const f = bag.pop();
  lastFilterId = f.id;
  return f;
}
function makeRect(filter) {
  return { filter, state: {}, x: 0, y: 0, w: 0, h: 0, tx: 0, ty: 0, tw: 0, th: 0, committed: false, commitT: 0, broken: false };
}
function setRectTarget(r, a, b) {
  const x1 = clamp(a.x, 0, W), x2 = clamp(b.x, 0, W), y1 = clamp(a.y, 0, H), y2 = clamp(b.y, 0, H);
  r.tx = Math.min(x1, x2) / W; r.ty = Math.min(y1, y2) / H;
  r.tw = Math.abs(x2 - x1) / W; r.th = Math.abs(y2 - y1) / H;
}
function snapRect(r) { r.x = r.tx; r.y = r.ty; r.w = r.tw; r.h = r.th; }
function commitRect() {
  const r = app.activeRect;
  app.activeRect = null;
  if (!r) return;
  snapRect(r);
  if (r.w * W < CONFIG.MIN_RECT_PX || r.h * H < CONFIG.MIN_RECT_PX) return;   // accidental pinch
  r.committed = true;
  r.commitT = performance.now() / 1000;
  app.rects.push(r);
  while (app.rects.length > CONFIG.MAX_RECTS) app.rects.shift();
  el.hint.classList.add('gone');
}

/* ---------- gesture recognition ---------- */
function handleHands(landmarks, v) {
  const hs = [];
  for (const lm of landmarks) {
    const palm = Math.hypot((lm[0].x - lm[9].x) * v.vw, (lm[0].y - lm[9].y) * v.vh) || 1;
    const gap = Math.hypot((lm[4].x - lm[8].x) * v.vw, (lm[4].y - lm[8].y) * v.vh);
    const p = toCanvas((lm[4].x + lm[8].x) / 2, (lm[4].y + lm[8].y) / 2, v);
    hs.push({ ratio: gap / palm, x: p.x, y: p.y, pinching: false, pts: lm.map((q) => toCanvas(q.x, q.y, v)) });
  }
  app.hands = hs;
  updateGesture(hs);
}
function updateGesture(hs) {
  const thr = G.phase === 'drawing' ? CONFIG.PINCH_HOLD : CONFIG.PINCH_START;
  for (const h of hs) h.pinching = h.ratio < thr;
  const pin = hs.filter((h) => h.pinching);
  app.pinchCount = pin.length;

  if (G.phase === 'idle') {
    if (pin.length >= 2 && !app.activeRect) {
      if (++G.on >= CONFIG.START_FRAMES) {
        const f = nextFilter();
        if (f) {
          app.activeRect = makeRect(f);
          setRectTarget(app.activeRect, pin[0], pin[1]);
          snapRect(app.activeRect);
          G.phase = 'drawing'; G.lost = 0;
        }
        G.on = 0;
      }
    } else G.on = 0;
  } else if (G.phase === 'drawing') {
    if (pin.length >= 2) {
      G.lost = 0;
      setRectTarget(app.activeRect, pin[0], pin[1]);
    } else if (++G.lost > CONFIG.LOST_GRACE) {
      commitRect();
      G.phase = 'release'; G.rel = 0;
    }
  } else if (G.phase === 'release') {
    // Wait for both hands to relax before the next gesture can begin.
    if (pin.length === 0) { if (++G.rel >= 4) G.phase = 'idle'; } else G.rel = 0;
  }
}

/* ---------- rendering ---------- */
function drawFilteredRect(v, r, t) {
  const x = r.x * W, y = r.y * H, w = r.w * W, h = r.h * H;
  if (w < 4 || h < 4 || r.broken) return;
  const maxPx = r.filter.maxPx || CONFIG.PROCESS_MAX_PX;
  const scale = Math.min(DPR, maxPx / Math.max(w, h));
  const pw = Math.max(2, Math.round(w * scale)), ph = Math.max(2, Math.round(h * scale));
  if (work.width !== pw || work.height !== ph) { work.width = pw; work.height = ph; }
  try {
    wctx.save();
    wctx.imageSmoothingEnabled = true;
    drawVideoRegion(wctx, v, x, y, w, h, pw, ph);
    r.filter.render({ c: wctx, src: work, w: pw, h: ph, t, scale: pw / w, state: r.state });
    wctx.restore();
    ctx.drawImage(work, 0, 0, pw, ph, x, y, w, h);
  } catch (err) {
    console.error('Effect "' + r.filter.id + '" failed', err);
    r.broken = true;
  }
}
function drawRectChrome(r, t) {
  const x = r.x * W, y = r.y * H, w = r.w * W, h = r.h * H;
  if (w < 2 || h < 2) return;
  const active = !r.committed;
  const flash = r.committed ? Math.max(0, 1 - (t - r.commitT) / 0.7) : 0;
  ctx.save();
  ctx.lineWidth = active ? 2.5 : 1.25 + flash * 1.5;
  ctx.strokeStyle = active ? '#ffc42e' : 'rgba(255,255,255,' + (0.34 + 0.66 * flash).toFixed(2) + ')';
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  if (active) {
    ctx.fillStyle = '#ffc42e';
    for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) ctx.fillRect(cx - 5, cy - 5, 10, 10);
  }
  if (CONFIG.SHOW_LABELS) {
    ctx.font = '600 12px "Avenir Next","Segoe UI",system-ui,sans-serif';
    const label = r.filter.name;
    const tw = ctx.measureText(label).width + 16, th = 22;
    const lx = clamp(x, 0, Math.max(0, W - tw));
    const ly = y >= th + 6 ? y - th - 5 : y + 6;
    ctx.fillStyle = 'rgba(12,14,18,.78)';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(lx, ly, tw, th, 11); else ctx.rect(lx, ly, tw, th);
    ctx.fill();
    ctx.fillStyle = active ? '#ffc42e' : '#f4f2ed';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, lx + 8, ly + th / 2 + 0.5);
  }
  ctx.restore();
}
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
function drawHands() {
  ctx.save();
  if (app.showSkeleton) {
    ctx.strokeStyle = 'rgba(255,255,255,.55)';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(255,255,255,.9)';
    for (const h of app.hands) {
      ctx.beginPath();
      for (const [a, b] of BONES) { ctx.moveTo(h.pts[a].x, h.pts[a].y); ctx.lineTo(h.pts[b].x, h.pts[b].y); }
      ctx.stroke();
      for (const p of h.pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, TAU); ctx.fill(); }
    }
  }
  if (CONFIG.SHOW_PINCH_POINTS) {
    const pinched = app.hands.filter((h) => h.pinching);
    if (pinched.length === 2) {
      ctx.strokeStyle = 'rgba(255,196,46,.7)';
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pinched[0].x, pinched[0].y); ctx.lineTo(pinched[1].x, pinched[1].y); ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const h of app.hands) {
      const rr = 7 + clamp(h.ratio, 0, 0.9) * 16;
      ctx.beginPath();
      ctx.arc(h.x, h.y, rr, 0, TAU);
      ctx.lineWidth = 2.5;
      if (h.pinching) { ctx.fillStyle = 'rgba(255,196,46,.9)'; ctx.fill(); ctx.strokeStyle = '#fff'; }
      else ctx.strokeStyle = 'rgba(255,255,255,.8)';
      ctx.stroke();
    }
  }
  ctx.restore();
}
function updateStatus() {
  if (G.phase === 'drawing' && app.activeRect) { setStatus('Drawing ' + app.activeRect.filter.name + '. Release to place it.', 'drawing'); return; }
  if (app.activeRect) { setStatus('Drawing ' + app.activeRect.filter.name + '. Let go to place it.', 'drawing'); return; }
  if (app.modelState === 'failed') { setStatus('Hand tracking is unavailable. Drag with the mouse to draw a box.', 'error'); return; }
  if (app.modelState !== 'ready') { setStatus('Loading hand tracking…', 'idle'); return; }
  const n = app.hands.length;
  if (n === 0) setStatus('Show both hands to the camera', 'idle');
  else if (n === 1) setStatus('Bring your other hand into view', 'idle');
  else if (app.pinchCount === 2) setStatus('Hold the pinch', 'ready');
  else if (app.pinchCount === 1) setStatus('Pinch with your other hand too', 'ready');
  else setStatus('Pinch thumb and index finger on both hands', 'ready');
}

function render(v, t) {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  drawVideoRegion(ctx, v, 0, 0, W, H, W, H);
  if (app.mode !== 'live') return;
  const list = app.activeRect ? app.rects.concat(app.activeRect) : app.rects;
  for (const r of list) {
    if (!r.committed) {
      const k = 0.45;
      r.x += (r.tx - r.x) * k; r.y += (r.ty - r.y) * k;
      r.w += (r.tw - r.w) * k; r.h += (r.th - r.h) * k;
    }
    drawFilteredRect(v, r, t);
  }
  for (const r of list) drawRectChrome(r, t);
  drawHands();
  updateStatus();
}

function loop(now) {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || !video.videoWidth) return;
  const v = getView();
  if (app.mode === 'live' && app.model && video.currentTime !== app.lastVideoTime) {
    app.lastVideoTime = video.currentTime;
    try {
      const res = app.model.detectForVideo(video, performance.now());
      handleHands(res.landmarks || [], v);
    } catch (err) { console.warn('Hand detection error', err); }
  }
  render(v, now / 1000);
}
requestAnimationFrame(loop);

/* ---------- start / modes ---------- */
async function startApp() {
  el.setup.hidden = true;
  if (app.modelState === 'loading' || app.modelState === 'idle') {
    el.loader.hidden = false;
    try { await app.modelPromise; } catch (e) { /* handled below */ }
    el.loader.hidden = true;
  }
  if (app.modelState === 'failed') toast('Hand tracking could not load, so you can draw boxes with the mouse instead. Check your connection and use Crop to retry.', 7000);
  app.mode = 'live';
  app.hands = []; app.pinchCount = 0;
  G.phase = 'idle'; G.on = 0;
  el.hud.hidden = false;
  lastStatus = '';
}
function recrop() {
  app.mode = 'setup';
  app.activeRect = null; G.phase = 'idle'; app.hands = [];
  el.hud.hidden = true;
  el.setup.hidden = false;
  if (app.modelState === 'failed') { app.modelState = 'idle'; app.modelPromise = loadModel(); }
}
function undo() { app.rects.pop(); }
function clearAll() { app.rects.length = 0; }
function toggleSkeleton() {
  app.showSkeleton = !app.showSkeleton;
  document.querySelector('[data-act="skeleton"]').setAttribute('aria-pressed', String(app.showSkeleton));
}
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
}

el.btnCam.addEventListener('click', openCamera);
el.btnStart.addEventListener('click', startApp);
el.btnReset.addEventListener('click', () => {
  crop.zoom = 1; crop.cx = 0.5; crop.cy = 0.5;
  el.zoom.value = '1'; el.zoomVal.textContent = '1.0×';
});
el.zoom.addEventListener('input', () => {
  crop.zoom = parseFloat(el.zoom.value);
  el.zoomVal.textContent = crop.zoom.toFixed(1) + '×';
});
el.mirror.addEventListener('change', () => { crop.mirror = el.mirror.checked; });
el.hud.addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]');
  if (!b) return;
  const act = b.dataset.act;
  if (act === 'undo') undo();
  else if (act === 'clear') clearAll();
  else if (act === 'skeleton') toggleSkeleton();
  else if (act === 'recrop') recrop();
  else if (act === 'fullscreen') toggleFullscreen();
});
window.addEventListener('keydown', (e) => {
  if (app.mode !== 'live' || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'z') undo();
  else if (k === 'c') clearAll();
  else if (k === 'h') toggleSkeleton();
  else if (k === 'f') toggleFullscreen();
});

/* ---------- pointer: pan the preview, or draw a box with the mouse as a fallback ---------- */
let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  if (app.mode === 'setup') {
    drag = { type: 'pan', x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  } else if (app.mode === 'live' && e.button === 0 && G.phase === 'idle' && !app.activeRect) {
    const f = nextFilter();
    if (!f) return;
    app.activeRect = makeRect(f);
    const p = { x: e.clientX, y: e.clientY };
    setRectTarget(app.activeRect, p, p);
    snapRect(app.activeRect);
    drag = { type: 'draw', x0: e.clientX, y0: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  }
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  if (drag.type === 'pan') {
    const v = getView();
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    crop.cx += (crop.mirror ? 1 : -1) * (dx / W) * v.sw / v.vw;
    crop.cy -= (dy / H) * v.sh / v.vh;
  } else if (drag.type === 'draw' && app.activeRect) {
    setRectTarget(app.activeRect, { x: drag.x0, y: drag.y0 }, { x: e.clientX, y: e.clientY });
  }
});
function endDrag() {
  if (drag && drag.type === 'draw') commitRect();
  drag = null;
}
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('wheel', (e) => {
  if (app.mode !== 'setup') return;
  e.preventDefault();
  crop.zoom = clamp(crop.zoom - e.deltaY * 0.002, 1, 4);
  el.zoom.value = String(crop.zoom);
  el.zoomVal.textContent = crop.zoom.toFixed(1) + '×';
}, { passive: false });

window.addEventListener('pagehide', () => {
  if (app.stream) app.stream.getTracks().forEach((tr) => tr.stop());
});
})();