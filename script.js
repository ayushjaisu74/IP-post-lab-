/* =========================================================
   SHARED UTILITIES
   ========================================================= */
const clamp = (v) => v < 0 ? 0 : v > 255 ? 255 : v;

function fileToImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

// Draw an HTMLImageElement onto a canvas, capping the longest side at maxDim.
function drawCapped(img, canvas, maxDim) {
  let { width: w, height: h } = img;
  if (Math.max(w, h) > maxDim) {
    const scale = maxDim / Math.max(w, h);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

function newCanvasLike(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

function putOn(canvas, imgData) {
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  canvas.getContext('2d').putImageData(imgData, 0, 0);
}

function cloneImageData(imgData) {
  return new ImageData(new Uint8ClampedArray(imgData.data), imgData.width, imgData.height);
}

function toGray(imgData) {
  const { data, width, height } = imgData;
  const g = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    g[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return g;
}

function grayToImageData(g, width, height) {
  const out = new ImageData(width, height);
  for (let p = 0, i = 0; p < g.length; p++, i += 4) {
    const v = clamp(Math.round(g[p]));
    out.data[i] = out.data[i + 1] = out.data[i + 2] = v;
    out.data[i + 3] = 255;
  }
  return out;
}

function readout(id, msg) {
  const el = document.getElementById(id);
  if (el) el.textContent = msg;
}

/* generic drag & drop wiring: fileInputId -> onFile(file) */
function wireDrop(dropSelector, inputEl, onFile) {
  const zone = document.querySelector(dropSelector);
  if (!zone) return;
  inputEl.addEventListener('change', (e) => {
    if (e.target.files[0]) onFile(e.target.files[0]);
  });
  ['dragover', 'dragenter'].forEach(ev => zone.addEventListener(ev, (e) => {
    e.preventDefault(); zone.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach(ev => zone.addEventListener(ev, (e) => {
    e.preventDefault(); zone.classList.remove('drag');
  }));
  zone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  });
}

/* generic 3x3 / 5x5 / 7x7 convolution on grayscale Float32Array */
function convolveGray(g, w, h, kernel, ksize, normalize) {
  const half = Math.floor(ksize / 2);
  const out = new Float32Array(w * h);
  let ksum = 0;
  if (normalize) { for (const k of kernel) ksum += k; if (ksum === 0) ksum = 1; }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0, ki = 0;
      for (let ky = -half; ky <= half; ky++) {
        const yy = Math.min(h - 1, Math.max(0, y + ky));
        for (let kx = -half; kx <= half; kx++) {
          const xx = Math.min(w - 1, Math.max(0, x + kx));
          acc += g[yy * w + xx] * kernel[ki++];
        }
      }
      out[y * w + x] = normalize ? acc / ksum : acc;
    }
  }
  return out;
}

function boxKernel(k) { return new Array(k * k).fill(1); }

function gaussianKernel(k, sigma) {
  const half = Math.floor(k / 2);
  const kernel = [];
  for (let y = -half; y <= half; y++) {
    for (let x = -half; x <= half; x++) {
      kernel.push(Math.exp(-(x * x + y * y) / (2 * sigma * sigma)));
    }
  }
  return kernel;
}

function medianFilterGray(g, w, h, k) {
  const half = Math.floor(k / 2);
  const out = new Float32Array(w * h);
  const win = new Array(k * k);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let n = 0;
      for (let ky = -half; ky <= half; ky++) {
        const yy = Math.min(h - 1, Math.max(0, y + ky));
        for (let kx = -half; kx <= half; kx++) {
          const xx = Math.min(w - 1, Math.max(0, x + kx));
          win[n++] = g[yy * w + xx];
        }
      }
      win.sort((a, b) => a - b);
      out[y * w + x] = win[Math.floor(n / 2)];
    }
  }
  return out;
}

function bilateralFilterGray(g, w, h, k, sigmaSpace, sigmaRange) {
  const half = Math.floor(k / 2);
  const out = new Float32Array(w * h);
  const spatial = [];
  for (let ky = -half; ky <= half; ky++) {
    for (let kx = -half; kx <= half; kx++) {
      spatial.push(Math.exp(-(kx * kx + ky * ky) / (2 * sigmaSpace * sigmaSpace)));
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const center = g[y * w + x];
      let acc = 0, wsum = 0, ki = 0;
      for (let ky = -half; ky <= half; ky++) {
        const yy = Math.min(h - 1, Math.max(0, y + ky));
        for (let kx = -half; kx <= half; kx++) {
          const xx = Math.min(w - 1, Math.max(0, x + kx));
          const val = g[yy * w + xx];
          const rangeW = Math.exp(-((val - center) ** 2) / (2 * sigmaRange * sigmaRange));
          const wgt = spatial[ki++] * rangeW;
          acc += val * wgt;
          wsum += wgt;
        }
      }
      out[y * w + x] = wsum > 0 ? acc / wsum : center;
    }
  }
  return out;
}

/* =========================================================
   P01 — FORMAT CONVERSION & PIXEL OPERATIONS
   ========================================================= */
(function initP01() {
  const canvas = document.getElementById('p01-canvas');
  const modeSel = document.getElementById('p01-mode');
  const constField = document.getElementById('p01-const-field');
  const secondField = document.getElementById('p01-second-field');
  const constRange = document.getElementById('p01-const');
  const constVal = document.getElementById('p01-const-val');
  const runBtn = document.getElementById('p01-run');
  let srcData = null, secondImg = null;

  constRange.addEventListener('input', () => constVal.textContent = constRange.value);
  modeSel.addEventListener('change', () => {
    const needsConst = !['not', 'add2'].includes(modeSel.value);
    constField.style.display = needsConst ? '' : 'none';
    secondField.style.display = modeSel.value === 'add2' ? '' : 'none';
  });

  wireDrop('[data-drop="p01"]', document.getElementById('p01-file'), async (file) => {
    const img = await fileToImage(file);
    srcData = drawCapped(img, canvas, 500);
    readout('p01-readout', `Loaded ${img.naturalWidth}×${img.naturalHeight}px — select an operation and run.`);
  });
  wireDrop('[data-drop="p01b"]', document.getElementById('p01-file2'), async (file) => {
    secondImg = await fileToImage(file);
    readout('p01-readout', 'Second image loaded. Run to blend.');
  });

  runBtn.addEventListener('click', () => {
    if (!srcData) { readout('p01-readout', 'Upload an image first.'); return; }
    const out = cloneImageData(srcData);
    const c = parseInt(constRange.value, 10);
    const mode = modeSel.value;
    const d = out.data;

    if (mode === 'gray') {
      for (let i = 0; i < d.length; i += 4) {
        const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        d[i] = d[i + 1] = d[i + 2] = g;
      }
    } else if (mode === 'add') {
      for (let i = 0; i < d.length; i += 4) { d[i] = clamp(d[i] + c); d[i+1] = clamp(d[i+1] + c); d[i+2] = clamp(d[i+2] + c); }
    } else if (mode === 'sub') {
      for (let i = 0; i < d.length; i += 4) { d[i] = clamp(d[i] - c); d[i+1] = clamp(d[i+1] - c); d[i+2] = clamp(d[i+2] - c); }
    } else if (mode === 'mul') {
      const factor = c / 64;
      for (let i = 0; i < d.length; i += 4) { d[i] = clamp(d[i] * factor); d[i+1] = clamp(d[i+1] * factor); d[i+2] = clamp(d[i+2] * factor); }
    } else if (mode === 'and') {
      for (let i = 0; i < d.length; i += 4) { d[i] &= c; d[i+1] &= c; d[i+2] &= c; }
    } else if (mode === 'or') {
      for (let i = 0; i < d.length; i += 4) { d[i] |= c; d[i+1] |= c; d[i+2] |= c; }
    } else if (mode === 'xor') {
      for (let i = 0; i < d.length; i += 4) { d[i] ^= c; d[i+1] ^= c; d[i+2] ^= c; }
    } else if (mode === 'not') {
      for (let i = 0; i < d.length; i += 4) { d[i] = 255 - d[i]; d[i+1] = 255 - d[i+1]; d[i+2] = 255 - d[i+2]; }
    } else if (mode === 'add2') {
      if (!secondImg) { readout('p01-readout', 'Upload a second image to blend.'); return; }
      const tmp = newCanvasLike(out.width, out.height);
      const tctx = tmp.getContext('2d');
      tctx.drawImage(secondImg, 0, 0, out.width, out.height);
      const d2 = tctx.getImageData(0, 0, out.width, out.height).data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp(d[i] * 0.5 + d2[i] * 0.5);
        d[i+1] = clamp(d[i+1] * 0.5 + d2[i+1] * 0.5);
        d[i+2] = clamp(d[i+2] * 0.5 + d2[i+2] * 0.5);
      }
    }
    putOn(canvas, out);
    readout('p01-readout', `Applied: ${modeSel.options[modeSel.selectedIndex].text}.`);
  });
})();

/* =========================================================
   P02 — GEOMETRIC TRANSFORMATIONS
   ========================================================= */
(function initP02() {
  const canvas = document.getElementById('p02-canvas');
  const modeSel = document.getElementById('p02-mode');
  const paramsBox = document.getElementById('p02-params');
  let img = null, baseW = 0, baseH = 0;

  const paramSets = {
    translate: [
      { id: 'tx', label: 'Shift X', min: -200, max: 200, val: 40 },
      { id: 'ty', label: 'Shift Y', min: -200, max: 200, val: 20 },
    ],
    rotate: [{ id: 'ang', label: 'Angle (°)', min: -180, max: 180, val: 30 }],
    scale: [{ id: 'sc', label: 'Scale ×100', min: 20, max: 250, val: 130 }],
    shear: [
      { id: 'shx', label: 'Shear X ×100', min: -80, max: 80, val: 25 },
      { id: 'shy', label: 'Shear Y ×100', min: -80, max: 80, val: 0 },
    ],
    reflect: [{ id: 'axis', label: 'Axis (0=H,1=V)', min: 0, max: 1, val: 0 }],
    crop: [
      { id: 'cx', label: 'Crop X %', min: 0, max: 80, val: 10 },
      { id: 'cy', label: 'Crop Y %', min: 0, max: 80, val: 10 },
      { id: 'cw', label: 'Crop W %', min: 10, max: 100, val: 60 },
      { id: 'ch', label: 'Crop H %', min: 10, max: 100, val: 60 },
    ],
  };

  function buildParams() {
    paramsBox.innerHTML = '';
    const set = paramSets[modeSel.value];
    set.forEach(p => {
      const field = document.createElement('div');
      field.className = 'field';
      field.innerHTML = `<label>${p.label} <span class="val" id="p02-${p.id}-val">${p.val}</span></label>
        <input type="range" id="p02-${p.id}" min="${p.min}" max="${p.max}" value="${p.val}">`;
      paramsBox.appendChild(field);
    });
    set.forEach(p => {
      const range = document.getElementById(`p02-${p.id}`);
      const label = document.getElementById(`p02-${p.id}-val`);
      range.addEventListener('input', () => { label.textContent = range.value; render(); });
    });
  }

  function getVal(id) { const el = document.getElementById(`p02-${id}`); return el ? parseFloat(el.value) : 0; }

  function render() {
    if (!img) return;
    const mode = modeSel.value;
    canvas.width = baseW; canvas.height = baseH;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, baseW, baseH);
    ctx.save();

    if (mode === 'translate') {
      ctx.translate(getVal('tx'), getVal('ty'));
      ctx.drawImage(img, 0, 0, baseW, baseH);
    } else if (mode === 'rotate') {
      const ang = getVal('ang') * Math.PI / 180;
      ctx.translate(baseW / 2, baseH / 2);
      ctx.rotate(ang);
      ctx.drawImage(img, -baseW / 2, -baseH / 2, baseW, baseH);
    } else if (mode === 'scale') {
      const s = getVal('sc') / 100;
      ctx.translate(baseW / 2, baseH / 2);
      ctx.scale(s, s);
      ctx.drawImage(img, -baseW / 2, -baseH / 2, baseW, baseH);
    } else if (mode === 'shear') {
      const shx = getVal('shx') / 100, shy = getVal('shy') / 100;
      ctx.translate(baseW / 2, baseH / 2);
      ctx.transform(1, shy, shx, 1, 0, 0);
      ctx.drawImage(img, -baseW / 2, -baseH / 2, baseW, baseH);
    } else if (mode === 'reflect') {
      const axis = getVal('axis');
      if (axis === 0) { ctx.translate(baseW, 0); ctx.scale(-1, 1); }
      else { ctx.translate(0, baseH); ctx.scale(1, -1); }
      ctx.drawImage(img, 0, 0, baseW, baseH);
    } else if (mode === 'crop') {
      const cx = getVal('cx') / 100 * baseW, cy = getVal('cy') / 100 * baseH;
      const cw = getVal('cw') / 100 * baseW, ch = getVal('ch') / 100 * baseH;
      ctx.restore();
      canvas.width = Math.max(1, Math.round(cw));
      canvas.height = Math.max(1, Math.round(ch));
      const ctx2 = canvas.getContext('2d');
      ctx2.drawImage(img, cx, cy, cw, ch, 0, 0, cw, ch);
      readout('p02-readout', `Cropped region ${Math.round(cw)}×${Math.round(ch)}px.`);
      return;
    }
    ctx.restore();
    readout('p02-readout', `Transform: ${modeSel.options[modeSel.selectedIndex].text}.`);
  }

  modeSel.addEventListener('change', () => { buildParams(); render(); });

  wireDrop('[data-drop="p02"]', document.getElementById('p02-file'), async (file) => {
    img = await fileToImage(file);
    const maxDim = 480;
    let w = img.naturalWidth, h = img.naturalHeight;
    if (Math.max(w, h) > maxDim) { const s = maxDim / Math.max(w, h); w = Math.round(w*s); h = Math.round(h*s); }
    baseW = w; baseH = h;
    buildParams();
    render();
  });

  buildParams();
})();

/* =========================================================
   P03 — HISTOGRAM EQ / SPATIAL FILTER / THRESHOLD
   ========================================================= */
(function initP03() {
  const canvas = document.getElementById('p03-canvas');
  const modeSel = document.getElementById('p03-mode');
  const threshField = document.getElementById('p03-thresh-field');
  const threshRange = document.getElementById('p03-thresh');
  const threshVal = document.getElementById('p03-thresh-val');
  let srcData = null;

  threshRange.addEventListener('input', () => threshVal.textContent = threshRange.value);
  modeSel.addEventListener('change', () => {
    threshField.style.display = modeSel.value === 'thresh' ? '' : 'none';
  });

  function drawHistogram(canvasId, g) {
    const hc = document.getElementById(canvasId);
    const ctx = hc.getContext('2d');
    ctx.clearRect(0, 0, hc.width, hc.height);
    const bins = new Array(256).fill(0);
    for (const v of g) bins[clamp(Math.round(v))]++;
    const max = Math.max(...bins);
    ctx.fillStyle = '#3DDC97';
    for (let x = 0; x < 256; x += 2) {
      const h = (bins[x] / max) * hc.height;
      ctx.fillRect((x / 256) * hc.width, hc.height - h, 1.5, h);
    }
  }

  wireDrop('[data-drop="p03"]', document.getElementById('p03-file'), async (file) => {
    const img = await fileToImage(file);
    srcData = drawCapped(img, canvas, 500);
    const g = toGray(srcData);
    drawHistogram('p03-hist-before', g);
    drawHistogram('p03-hist-after', g);
    readout('p03-readout', `Loaded ${img.naturalWidth}×${img.naturalHeight}px — select an operation and run.`);
  });

  document.getElementById('p03-run').addEventListener('click', () => {
    if (!srcData) { readout('p03-readout', 'Upload an image first.'); return; }
    const { width: w, height: h } = srcData;
    let g = toGray(srcData);
    drawHistogram('p03-hist-before', g);
    const mode = modeSel.value;

    if (mode === 'eq') {
      const hist = new Array(256).fill(0);
      for (const v of g) hist[clamp(Math.round(v))]++;
      const cdf = new Array(256).fill(0);
      let running = 0;
      for (let i = 0; i < 256; i++) { running += hist[i]; cdf[i] = running; }
      const cdfMin = cdf.find(v => v > 0) || 0;
      const total = w * h;
      const lut = new Array(256).fill(0).map((_, i) => Math.round(((cdf[i] - cdfMin) / (total - cdfMin)) * 255));
      g = g.map(v => lut[clamp(Math.round(v))]);
      readout('p03-readout', 'Histogram equalization applied (global CDF mapping).');
    } else if (mode === 'smooth') {
      g = convolveGray(g, w, h, boxKernel(3), 3, true);
      readout('p03-readout', 'Spatial smoothing applied (3×3 box filter).');
    } else if (mode === 'sharpen') {
      const lap = [0, -1, 0, -1, 5, -1, 0, -1, 0];
      const filtered = convolveGray(g, w, h, lap, 3, false);
      g = filtered.map(v => clamp(v));
      readout('p03-readout', 'Spatial sharpening applied (Laplacian kernel).');
    } else if (mode === 'thresh') {
      const t = parseInt(threshRange.value, 10);
      g = g.map(v => v >= t ? 255 : 0);
      readout('p03-readout', `Global threshold applied at T = ${t}.`);
    }
    putOn(canvas, grayToImageData(g, w, h));
    drawHistogram('p03-hist-after', g);
  });
})();

/* =========================================================
   P04 — AVERAGING / GAUSSIAN / MEDIAN / BILATERAL FILTERS
   ========================================================= */
(function initP04() {
  const kRange = document.getElementById('p04-k');
  const kVal = document.getElementById('p04-k-val');
  kRange.addEventListener('input', () => kVal.textContent = kRange.value);
  let srcData = null;
  const holder = document.createElement('canvas');

  wireDrop('[data-drop="p04"]', document.getElementById('p04-file'), async (file) => {
    const img = await fileToImage(file);
    srcData = drawCapped(img, holder, 260);
    readout('p04-readout', `Loaded ${img.naturalWidth}×${img.naturalHeight}px (processing at ${holder.width}×${holder.height} for speed). Click run.`);
  });

  document.getElementById('p04-run').addEventListener('click', () => {
    if (!srcData) { readout('p04-readout', 'Upload an image first.'); return; }
    readout('p04-readout', 'Processing…');
    setTimeout(() => {
      const { width: w, height: h } = srcData;
      const g = toGray(srcData);
      const k = parseInt(kRange.value, 10);

      const avg = convolveGray(g, w, h, boxKernel(k), k, true);
      putOn(document.getElementById('p04-avg'), grayToImageData(avg, w, h));

      const gauss = convolveGray(g, w, h, gaussianKernel(k, k / 3), k, true);
      putOn(document.getElementById('p04-gauss'), grayToImageData(gauss, w, h));

      const med = medianFilterGray(g, w, h, k);
      putOn(document.getElementById('p04-med'), grayToImageData(med, w, h));

      const bil = bilateralFilterGray(g, w, h, k, k / 2, 30);
      putOn(document.getElementById('p04-bil'), grayToImageData(bil, w, h));

      readout('p04-readout', `All four filters applied with kernel size ${k}×${k}.`);
    }, 30);
  });
})();

/* =========================================================
   P05 — INPAINTING (Telea-style / Navier-Stokes-style)
   ========================================================= */
(function initP05() {
  const srcCanvas = document.getElementById('p05-src');
  const outCanvas = document.getElementById('p05-out');
  const brushRange = document.getElementById('p05-brush');
  const brushVal = document.getElementById('p05-brush-val');
  let baseData = null, mask = null, w = 0, h = 0, drawing = false;

  brushRange.addEventListener('input', () => brushVal.textContent = brushRange.value);

  wireDrop('[data-drop="p05"]', document.getElementById('p05-file'), async (file) => {
    const img = await fileToImage(file);
    baseData = drawCapped(img, srcCanvas, 320);
    w = srcCanvas.width; h = srcCanvas.height;
    mask = new Uint8Array(w * h);
    outCanvas.width = w; outCanvas.height = h;
    outCanvas.getContext('2d').clearRect(0, 0, w, h);
    readout('p05-readout', `Loaded ${img.naturalWidth}×${img.naturalHeight}px — scribble on the left canvas to mark damage, then inpaint.`);
    renderMaskPreview();
  });

  function renderMaskPreview() {
    const ctx = srcCanvas.getContext('2d');
    ctx.putImageData(baseData, 0, 0);
    ctx.fillStyle = 'rgba(232,106,92,0.55)';
    for (let p = 0; p < mask.length; p++) {
      if (mask[p]) {
        const x = p % w, y = (p / w) | 0;
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }

  function paintAt(x, y, r) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx*dx + dy*dy > r*r) continue;
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        mask[yy * w + xx] = 1;
      }
    }
  }

  function canvasPos(e) {
    const rect = srcCanvas.getBoundingClientRect();
    const scaleX = srcCanvas.width / rect.width, scaleY = srcCanvas.height / rect.height;
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return { x: Math.round(cx * scaleX), y: Math.round(cy * scaleY) };
  }

  ['mousedown','touchstart'].forEach(ev => srcCanvas.addEventListener(ev, (e) => {
    if (!baseData) return;
    drawing = true;
    const { x, y } = canvasPos(e);
    paintAt(x, y, parseInt(brushRange.value, 10) / 2);
    renderMaskPreview();
  }));
  ['mousemove','touchmove'].forEach(ev => srcCanvas.addEventListener(ev, (e) => {
    if (!drawing || !baseData) return;
    e.preventDefault();
    const { x, y } = canvasPos(e);
    paintAt(x, y, parseInt(brushRange.value, 10) / 2);
    renderMaskPreview();
  }));
  ['mouseup','mouseleave','touchend'].forEach(ev => srcCanvas.addEventListener(ev, () => drawing = false));

  document.getElementById('p05-clear').addEventListener('click', () => {
    if (!baseData) return;
    mask.fill(0);
    renderMaskPreview();
    readout('p05-readout', 'Mask cleared.');
  });

  // Diffusion-based inpainting: iteratively replace masked pixels with the
  // average of their unmasked (or already-updated) neighbours. The "Telea"
  // mode weights by inverse distance to the mask boundary (fast-marching
  // style, outside-in); the "NS" mode is a uniform isotropic diffusion,
  // approximating the smoothing behaviour of the Navier-Stokes method.
  function inpaint(method) {
    const data = new Float32Array(baseData.data); // rgba
    const m = new Uint8Array(mask);
    const iterations = method === 'telea' ? 60 : 120;

    for (let it = 0; it < iterations; it++) {
      let changed = false;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const p = y * w + x;
          if (!m[p]) continue;
          let rSum=0,gSum=0,bSum=0,wsum=0;
          const neighbors = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]];
          for (const [dx,dy] of neighbors) {
            const xx=x+dx, yy=y+dy;
            if (xx<0||yy<0||xx>=w||yy>=h) continue;
            const np = yy*w+xx;
            if (m[np]) continue; // only pull from resolved pixels
            const idx = np*4;
            const weight = method === 'telea' ? 1/Math.sqrt(dx*dx+dy*dy) : 1;
            rSum += data[idx]*weight; gSum += data[idx+1]*weight; bSum += data[idx+2]*weight; wsum += weight;
          }
          if (wsum > 0) {
            const idx = p*4;
            data[idx] = rSum/wsum; data[idx+1] = gSum/wsum; data[idx+2] = bSum/wsum;
            m[p] = 0; // mark resolved for this pass
            changed = true;
          }
        }
      }
      if (!changed) break;
    }
    const out = new ImageData(w, h);
    for (let i = 0; i < data.length; i++) out.data[i] = i % 4 === 3 ? 255 : clamp(Math.round(data[i]));
    return out;
  }

  document.getElementById('p05-run').addEventListener('click', () => {
    if (!baseData) { readout('p05-readout', 'Upload an image first.'); return; }
    if (!mask.some(v => v)) { readout('p05-readout', 'Scribble a mask over the region to remove first.'); return; }
    readout('p05-readout', 'Inpainting…');
    setTimeout(() => {
      const method = document.getElementById('p05-method').value;
      const result = inpaint(method);
      putOn(outCanvas, result);
      readout('p05-readout', `Reconstructed with ${method === 'telea' ? 'Telea-style fast marching' : 'Navier–Stokes-style diffusion'} fill.`);
    }, 30);
  });
})();

/* =========================================================
   P06 — LOSSLESS COMPRESSION
   ========================================================= */
(function initP06() {
  const canvas = document.getElementById('p06-canvas');
  let originalBytes = 0, imgData = null;

  function fmtBytes(n) {
    if (n < 1024) return n + ' B';
    if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
    return (n/1024/1024).toFixed(2) + ' MB';
  }

  // Simple lossless RLE over raw RGBA bytes (run-length encode identical
  // consecutive byte values — a minimal, verifiably lossless scheme).
  function rleEncode(bytes) {
    let count = 0;
    let i = 0;
    while (i < bytes.length) {
      let run = 1;
      while (i + run < bytes.length && bytes[i + run] === bytes[i] && run < 255) run++;
      count += 2; // [value, runlength] pair
      i += run;
    }
    return count;
  }

  wireDrop('[data-drop="p06"]', document.getElementById('p06-file'), async (file) => {
    originalBytes = file.size;
    const img = await fileToImage(file);
    imgData = drawCapped(img, canvas, 600);
    document.getElementById('p06-orig').textContent = fmtBytes(originalBytes);
    document.getElementById('p06-png').textContent = '—';
    document.getElementById('p06-rle').textContent = '—';
    readout('p06-readout', `Loaded ${img.naturalWidth}×${img.naturalHeight}px — click compress to compare.`);
  });

  document.getElementById('p06-run').addEventListener('click', () => {
    if (!imgData) { readout('p06-readout', 'Upload an image first.'); return; }
    canvas.toBlob((blob) => {
      const pngSize = blob.size;
      document.getElementById('p06-png').textContent = fmtBytes(pngSize);
      document.getElementById('p06-png-ratio').textContent = (originalBytes / pngSize).toFixed(2) + '×';

      const rleBytes = rleEncode(imgData.data);
      document.getElementById('p06-rle').textContent = fmtBytes(rleBytes);
      document.getElementById('p06-rle-ratio').textContent = (originalBytes / rleBytes).toFixed(2) + '×';

      readout('p06-readout', 'Both re-encodings are lossless — decoding either reproduces the exact pixel data shown above.');
    }, 'image/png');
  });
})();

/* =========================================================
   P07 — MORPHOLOGICAL OPERATIONS
   ========================================================= */
(function initP07() {
  const threshRange = document.getElementById('p07-thresh');
  const threshVal = document.getElementById('p07-thresh-val');
  const iterRange = document.getElementById('p07-iter');
  const iterVal = document.getElementById('p07-iter-val');
  threshRange.addEventListener('input', () => threshVal.textContent = threshRange.value);
  iterRange.addEventListener('input', () => iterVal.textContent = iterRange.value);

  let srcData = null;
  const holder = document.createElement('canvas');

  wireDrop('[data-drop="p07"]', document.getElementById('p07-file'), async (file) => {
    const img = await fileToImage(file);
    srcData = drawCapped(img, holder, 320);
    readout('p07-readout', `Loaded ${img.naturalWidth}×${img.naturalHeight}px. Click run.`);
  });

  function binarize(g, w, h, t) {
    const bin = new Uint8Array(w * h);
    for (let i = 0; i < g.length; i++) bin[i] = g[i] >= t ? 1 : 0;
    return bin;
  }

  function erode(bin, w, h) {
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let all = 1;
      for (let dy = -1; dy <= 1 && all; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x+dx, yy = y+dy;
        if (xx<0||yy<0||xx>=w||yy>=h || !bin[yy*w+xx]) { all = 0; break; }
      }
      out[y*w+x] = all;
    }
    return out;
  }
  function dilate(bin, w, h) {
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let any = 0;
      for (let dy = -1; dy <= 1 && !any; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x+dx, yy = y+dy;
        if (xx>=0&&yy>=0&&xx<w&&yy<h && bin[yy*w+xx]) { any = 1; break; }
      }
      out[y*w+x] = any;
    }
    return out;
  }
  function binToImageData(bin, w, h) {
    const out = new ImageData(w, h);
    for (let p = 0, i = 0; p < bin.length; p++, i += 4) {
      const v = bin[p] ? 255 : 0;
      out.data[i]=out.data[i+1]=out.data[i+2]=v; out.data[i+3]=255;
    }
    return out;
  }

  document.getElementById('p07-run').addEventListener('click', () => {
    if (!srcData) { readout('p07-readout', 'Upload an image first.'); return; }
    const { width: w, height: h } = srcData;
    const g = toGray(srcData);
    const t = parseInt(threshRange.value, 10);
    const iters = parseInt(iterRange.value, 10);
    let bin = binarize(g, w, h, t);

    let er = bin;
    for (let i = 0; i < iters; i++) er = erode(er, w, h);
    putOn(document.getElementById('p07-erode'), binToImageData(er, w, h));

    let di = bin;
    for (let i = 0; i < iters; i++) di = dilate(di, w, h);
    putOn(document.getElementById('p07-dilate'), binToImageData(di, w, h));

    let op = bin;
    for (let i = 0; i < iters; i++) op = erode(op, w, h);
    for (let i = 0; i < iters; i++) op = dilate(op, w, h);
    putOn(document.getElementById('p07-open'), binToImageData(op, w, h));

    let cl = bin;
    for (let i = 0; i < iters; i++) cl = dilate(cl, w, h);
    for (let i = 0; i < iters; i++) cl = erode(cl, w, h);
    putOn(document.getElementById('p07-close'), binToImageData(cl, w, h));

    readout('p07-readout', `Binarized at T=${t}, ${iters} iteration(s) with a 3×3 structuring element.`);
  });
})();

/* =========================================================
   P08 — CORRELATION / TEMPLATE MATCHING
   ========================================================= */
(function initP08() {
  const srcCanvas = document.getElementById('p08-src');
  const outCanvas = document.getElementById('p08-out');
  let baseData = null, w = 0, h = 0;
  let dragStart = null, sel = null;

  wireDrop('[data-drop="p08"]', document.getElementById('p08-file'), async (file) => {
    const img = await fileToImage(file);
    baseData = drawCapped(img, srcCanvas, 300);
    w = srcCanvas.width; h = srcCanvas.height;
    sel = null;
    readout('p08-readout', `Loaded ${img.naturalWidth}×${img.naturalHeight}px — drag a small square on the canvas to pick a template.`);
  });

  function canvasPos(e) {
    const rect = srcCanvas.getBoundingClientRect();
    const scaleX = srcCanvas.width / rect.width, scaleY = srcCanvas.height / rect.height;
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY)
    };
  }

  function redrawSelection() {
    const ctx = srcCanvas.getContext('2d');
    ctx.putImageData(baseData, 0, 0);
    if (sel) {
      ctx.strokeStyle = '#E8A33D';
      ctx.lineWidth = 2;
      ctx.strokeRect(sel.x, sel.y, sel.w, sel.h);
    }
  }

  srcCanvas.addEventListener('mousedown', (e) => {
    if (!baseData) return;
    dragStart = canvasPos(e);
  });
  srcCanvas.addEventListener('mousemove', (e) => {
    if (!dragStart || !baseData) return;
    const p = canvasPos(e);
    sel = {
      x: Math.min(dragStart.x, p.x), y: Math.min(dragStart.y, p.y),
      w: Math.abs(p.x - dragStart.x), h: Math.abs(p.y - dragStart.y)
    };
    redrawSelection();
  });
  window.addEventListener('mouseup', () => { dragStart = null; });

  document.getElementById('p08-run').addEventListener('click', () => {
    if (!baseData) { readout('p08-readout', 'Upload an image first.'); return; }
    if (!sel || sel.w < 6 || sel.h < 6) { readout('p08-readout', 'Drag a template selection (at least ~6×6px) on the source image first.'); return; }
    readout('p08-readout', 'Computing normalized cross-correlation…');
    setTimeout(() => {
      const g = toGray(baseData);
      const tw = Math.min(sel.w, 80), th = Math.min(sel.h, 80);
      const template = new Float32Array(tw * th);
      let tMean = 0;
      for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
        const v = g[(sel.y+y)*w + (sel.x+x)];
        template[y*tw+x] = v; tMean += v;
      }
      tMean /= template.length;
      let tNorm = 0;
      const tc = template.map(v => { const d = v - tMean; tNorm += d*d; return d; });
      tNorm = Math.sqrt(tNorm) || 1;

      const outW = w - tw + 1, outH = h - th + 1;
      const corr = new Float32Array(Math.max(1,outW) * Math.max(1,outH));
      let maxC = -Infinity;
      for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
          let mean = 0;
          for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) mean += g[(y+ty)*w+(x+tx)];
          mean /= (tw*th);
          let num = 0, denom = 0;
          for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
            const d = g[(y+ty)*w+(x+tx)] - mean;
            num += d * tc[ty*tw+tx];
            denom += d*d;
          }
          const score = num / ((Math.sqrt(denom) || 1) * tNorm);
          corr[y*outW+x] = score;
          if (score > maxC) maxC = score;
        }
      }
      // normalize to 0..255 for display
      let minC = Infinity;
      for (const v of corr) if (v < minC) minC = v;
      const range = (maxC - minC) || 1;
      const heat = new Float32Array(outW*outH);
      for (let i = 0; i < corr.length; i++) heat[i] = ((corr[i]-minC)/range) * 255;
      putOn(outCanvas, grayToImageData(heat, outW, outH));
      readout('p08-readout', `Correlation map computed (${outW}×${outH}). Peak NCC score: ${maxC.toFixed(3)} — brightest point marks the best match.`);
    }, 30);
  });
})();

/* =========================================================
   RAIL MOBILE TOGGLE + SCROLLSPY
   ========================================================= */
(function initRail() {
  const toggle = document.getElementById('railToggle');
  const list = document.querySelector('.rail-list');
  toggle.addEventListener('click', () => list.classList.toggle('open'));
  list.querySelectorAll('a').forEach(a => a.addEventListener('click', () => list.classList.remove('open')));

  const links = list.querySelectorAll('a');
  const sections = Array.from(links).map(a => document.querySelector(a.getAttribute('href')));
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(l => l.style.borderLeftColor = 'transparent');
        const active = list.querySelector(`a[href="#${entry.target.id}"]`);
        if (active) active.style.borderLeftColor = 'var(--green)';
      }
    });
  }, { rootMargin: '-40% 0px -50% 0px' });
  sections.forEach(s => s && obs.observe(s));
})();
