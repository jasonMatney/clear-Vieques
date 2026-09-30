// ClearVieques — spectral wave cascades (3 x 256^2 FFT patches) driven entirely on the GPU.
(function () {
  'use strict';
  const CV = window.CV;
  const N = 256, CASCADES = 3, GRAV = 9.81;
  const WIND_S_PEAK = 4, WIND_S_HIGH = 1.3;   // wind-sea directional spread: cos^2s lobe, s = 4 at the peak widening to ~1.3 for wavelets (see waveInit)
  const T_LOOP = 256; // s — quantised dispersion makes the pattern loop exactly (no float32 phase drift)

  function dirNorm(s) {
    let sum = 0; const n = 720;
    for (let i = 0; i < n; i++) { const th = -Math.PI + (i + 0.5) * 2 * Math.PI / n; sum += Math.pow(Math.max(Math.cos(th / 2), 0), 2 * s); }
    return 1 / (sum * 2 * Math.PI / n);
  }
  function jonswapM0(wp, gamma) { // moment m0 of a JONSWAP spectrum with alpha = 1
    let m0 = 0; const w0 = 0.05, w1 = 45, n = 6000, dw = (w1 - w0) / n;
    for (let i = 0; i < n; i++) {
      const w = w0 + (i + 0.5) * dw, sigma = w <= wp ? 0.07 : 0.09, r = Math.exp(-((w - wp) ** 2) / (2 * sigma * sigma * wp * wp));
      m0 += GRAV * GRAV * Math.pow(w, -5) * Math.exp(-1.25 * Math.pow(wp / w, 4)) * Math.pow(gamma, r) * dw;
    }
    return m0;
  }

  CV.Waves = class Waves {
    constructor(gpu, mip) {
      this.gpu = gpu; this.mip = mip; this.N = N;
      this.params = { wind: 5, energy: 0.35, windDirDeg: 250, swellDirDeg: 340, depth: 6, choppy: 0.85, choppySign: 1, foamGain: 2.0, foamThreshold: 0.62 };
      this.dirty = true; this.cur = 0; this.frame = 0;
      this.L = [250, 34, 4.7]; this.kb = [2.0, 15.0];
      this.ready = this.init();
    }
    // Directions are compass bearings the waves come FROM (deg, clockwise from north) -> direction of travel in (x east, z south).
    static travelAngle(fromDeg) {
      const to = (fromDeg + 180) * CV.D2R; // bearing toward which they travel
      return Math.atan2(-Math.cos(to), Math.sin(to)); // dz = -cos(bearing), dx = sin(bearing)
    }
    set(p) { Object.assign(this.params, p); this.dirty = true; this.mssDirty = true; }
    // Mean-square slope of each cascade (used by the shader to know how much slope variance filtering removes).
    mss() { if (this.mssDirty || !this._mss) { this._mss = this.expected(4).map(e => e.mss); this.mssDirty = false; } return this._mss; }

    async init() {
      const gpu = this.gpu, dev = gpu.device;
      const mk = async (label, code) => CV.shader(gpu, label, code);
      const [mInit, mEvolve, mRows, mCols, mAsm] = await Promise.all([
        mk('waveInit', CV.wgsl.waveInit), mk('waveEvolve', CV.wgsl.waveEvolve), mk('fftRows', CV.wgsl.waveFftRows),
        mk('fftCols', CV.wgsl.waveFftCols), mk('waveAssemble', CV.wgsl.waveAssemble)]);
      const pipe = (label, module) => dev.createComputePipeline({ label, layout: 'auto', compute: { module, entryPoint: 'main' } });
      this.pInit = pipe('waveInit', mInit); this.pEvolve = pipe('waveEvolve', mEvolve);
      this.pRows = pipe('fftRows', mRows); this.pCols = pipe('fftCols', mCols); this.pAsm = pipe('waveAssemble', mAsm);

      const S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_DST;
      this.uBuf = CV.buffer(gpu, 9 * 16, GPUBufferUsage.UNIFORM | C, 'waveU');
      this.h0 = CV.buffer(gpu, CASCADES * N * N * 8, S, 'h0');
      this.specA = CV.buffer(gpu, CASCADES * N * N * 16, S, 'specA');
      this.specB = CV.buffer(gpu, CASCADES * N * N * 16, S, 'specB');

      const levels = CV.mipCount(N, N), usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
      const tex = (label) => dev.createTexture({ label, size: [N, N, CASCADES], format: 'rgba16float', mipLevelCount: levels, usage });
      this.dispTex = tex('waveDisp'); this.slopeTex = [tex('waveSlopeA'), tex('waveSlopeB')];
      const arr = (t, mipLevelCount, base = 0) => t.createView({ dimension: '2d-array', baseMipLevel: base, mipLevelCount });
      this.dispView = arr(this.dispTex, levels); this.slopeView = this.slopeTex.map(t => arr(t, levels));
      const disp0 = arr(this.dispTex, 1), slope0 = this.slopeTex.map(t => arr(t, 1));

      const bg = (pipeline, entries) => dev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map((resource, binding) => ({ binding, resource })) });
      this.bgInit = bg(this.pInit, [{ buffer: this.uBuf }, { buffer: this.h0 }]);
      this.bgEvolve = bg(this.pEvolve, [{ buffer: this.uBuf }, { buffer: this.h0 }, { buffer: this.specA }, { buffer: this.specB }]);
      this.bgRows = bg(this.pRows, [{ buffer: this.specA }, { buffer: this.specB }]);
      this.bgCols = bg(this.pCols, [{ buffer: this.specA }, { buffer: this.specB }]);
      this.bgAsm = [0, 1].map(cur => bg(this.pAsm, [{ buffer: this.uBuf }, { buffer: this.specA }, { buffer: this.specB },
        this.slopeView[1 - cur], disp0, slope0[cur]]));
      await this.mip.ready;
      this.mipDisp = this.mip.makePasses(this.dispTex, N, N, CASCADES);
      this.mipSlope = this.slopeTex.map(t => this.mip.makePasses(t, N, N, CASCADES));
    }

    uniformData(time, dt) {
      const p = this.params, d = new Float32Array(9 * 4);
      const U = Math.max(p.wind, 0.4), E = p.energy;
      // wind sea: fetch-limited JONSWAP (fetch grows with the wave-energy control)
      const F = 1500 + 9000 * E;
      const alphaW = CV.clamp(0.076 * Math.pow(U * U / (F * GRAV), 0.22), 0.003, 0.05);
      const wpW = 22 * Math.pow(GRAV * GRAV / (U * F), 1 / 3);
      // swell: narrow JONSWAP scaled to a target significant height
      const hsS = 0.04 + 0.28 * E, wpS = 2 * Math.PI / 7.5, gS = 6;
      const alphaS = (hsS / 4) ** 2 / jonswapM0(wpS, gS);
      this.info = { hsWind: 4 * Math.sqrt(alphaW * jonswapM0(wpW, 3.3)), wpWind: wpW, hsSwell: hsS, fetch: F, alphaW };
      const w0 = 2 * Math.PI / T_LOOP;
      const choppy = Math.min(1.55, p.choppy + 0.055 * Math.max(U - 3, 0)); // crest sharpening grows with wind
      d.set([time % T_LOOP, choppy, p.depth, w0], 0);
      const wc = CV.smoothstep(3.5, 9.0, U); // whitecap coverage grows steeply with wind (breaking starts ~4 m/s)
      d.set([p.foamGain * wc, Math.exp(-Math.min(dt, 0.1) / 2.4), p.foamThreshold, p.choppySign], 4);
      d.set([this.L[0], 0, this.kb[0], 11], 8);
      d.set([this.L[1], this.kb[0], this.kb[1], 23], 12);
      d.set([this.L[2], this.kb[1], 1e9, 37], 16);
      d.set([alphaW, wpW, 3.3, CV.Waves.travelAngle(p.windDirDeg)], 20);
      d.set([WIND_S_PEAK, WIND_S_HIGH, wpW * wpW / GRAV, 0], 24);
      d.set([alphaS, wpS, gS, CV.Waves.travelAngle(p.swellDirDeg)], 28);
      d.set([30, dirNorm(30), 0, 0], 32);
      return d;
    }


    // ---------------------------------------------------------------------------------- diagnostics (regression check of the FFT normalisation)
    // Analytic expectation: variance of height and mean-square slope per cascade, summing the same spectrum the GPU uses.
    expected(stride = 1) {
      const p = this.params, U = Math.max(p.wind, 0.4), E = p.energy, F = 1500 + 9000 * E;
      const alphaW = CV.clamp(0.076 * Math.pow(U * U / (F * GRAV), 0.22), 0.003, 0.05), wpW = 22 * Math.pow(GRAV * GRAV / (U * F), 1 / 3);
      const hsS = 0.04 + 0.28 * E, wpS = 2 * Math.PI / 7.5, aS = (hsS / 4) ** 2 / jonswapM0(wpS, 6);
      const disp = (k) => Math.sqrt(GRAV * k * Math.tanh(Math.min(k * p.depth, 40)) * (1 + (k / 363) ** 2));
      const jon = (w, a, wp, g) => { if (w < 1e-3) return 0; const sg = w <= wp ? 0.07 : 0.09, r = Math.exp(-((w - wp) ** 2) / (2 * sg * sg * wp * wp)); return a * GRAV * GRAV * Math.pow(w, -5) * Math.exp(-1.25 * Math.pow(wp / w, 4)) * Math.pow(g, r); };
      const dirW = CV.Waves.travelAngle(p.windDirDeg), dirS = CV.Waves.travelAngle(p.swellDirDeg), nS = dirNorm(30), kp = wpW * wpW / GRAV;
      const spread = (th, d, s, n) => { const x = th - d, dd = Math.atan2(Math.sin(x), Math.cos(x)); return n * Math.pow(Math.max(Math.cos(0.5 * dd), 0), 2 * s); };
      const sm = (a, b, x) => { const t = CV.clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
      const windSpread = (th, k) => { const t = CV.clamp((Math.log2(Math.max(k, 1e-3) / Math.max(kp, 1e-3)) - 1) / 3, 0, 1), s = CV.mix(WIND_S_PEAK, WIND_S_HIGH, t * t * (3 - 2 * t)); return spread(th, dirW, s, Math.sqrt(s + 0.25) * 0.28209479); };
      const out = [];
      for (let c = 0; c < 3; c++) {
        const L = this.L[c], kLo = c === 0 ? 0 : this.kb[c - 1], kHi = c === 2 ? 1e9 : this.kb[c], dk = 2 * Math.PI / L, kn = Math.PI * N / L;
        let varH = 0, mss = 0;
        for (let j = 0; j < N; j += stride) for (let i = 0; i < N; i += stride) {
          const nx = i >= N / 2 ? i - N : i, nz = j >= N / 2 ? j - N : j;
          if (nx === -N / 2 || nz === -N / 2 || (nx === 0 && nz === 0)) continue;
          const kx = nx * dk, kz = nz * dk, k = Math.hypot(kx, kz), th = Math.atan2(kz, kx), w = disp(k);
          const e = k * 0.002 + 1e-5, cg = (disp(k + e) - disp(Math.max(k - e, 1e-6))) / (2 * e);
          let psi = (jon(w, alphaW, wpW, 3.3) * windSpread(th, k) + jon(w, aS, wpS, 6) * spread(th, dirS, 30, nS)) * cg / k;
          let win = 1; if (kLo > 0) win *= sm(kLo * 0.85, kLo * 1.15, k); if (kHi < 1e8) win *= 1 - sm(kHi * 0.85, kHi * 1.15, k);
          win *= 1 - sm(0.72 * kn, 0.94 * kn, Math.max(Math.abs(kx), Math.abs(kz)));
          psi *= win; varH += psi * dk * dk * stride * stride; mss += k * k * psi * dk * dk * stride * stride;
        }
        out.push({ cascade: c, L, rmsH: Math.sqrt(varH), mss });
      }
      return out;
    }
    async readStats() {
      const gpu = this.gpu, dev = gpu.device, cur = this.cur, bpr = N * 8, size = bpr * N;
      const f16 = (h) => { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : e === 31 ? NaN : s * (1 + m / 1024) * 2 ** (e - 15); };
      const res = [];
      for (let c = 0; c < CASCADES; c++) {
        const bufs = [this.dispTex, this.slopeTex[cur]].map(() => dev.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
        const enc = dev.createCommandEncoder();
        [this.dispTex, this.slopeTex[cur]].forEach((t, i) => enc.copyTextureToBuffer({ texture: t, origin: [0, 0, c] }, { buffer: bufs[i], bytesPerRow: bpr }, [N, N, 1]));
        dev.queue.submit([enc.finish()]);
        await Promise.all(bufs.map(b => b.mapAsync(GPUMapMode.READ)));
        const d = new Uint16Array(bufs[0].getMappedRange()), s = new Uint16Array(bufs[1].getMappedRange());
        let sh = 0, sh2 = 0, ss = 0, sJ = 0, sJmin = 9, foam = 0, dx2 = 0; const n = N * N;
        for (let i = 0; i < n; i++) {
          const h = f16(d[i * 4 + 1]), sx = f16(s[i * 4]), sz = f16(s[i * 4 + 1]), J = f16(s[i * 4 + 2]), fo = f16(s[i * 4 + 3]), dx = f16(d[i * 4]);
          sh += h; sh2 += h * h; ss += sx * sx + sz * sz; sJ += J; sJmin = Math.min(sJmin, J); foam += fo; dx2 += dx * dx;
        }
        bufs.forEach(b => { b.unmap(); b.destroy(); });
        res.push({ cascade: c, rmsH: Math.sqrt(sh2 / n - (sh / n) ** 2), mss: ss / n, rmsDispX: Math.sqrt(dx2 / n), meanJ: sJ / n, minJ: sJmin, foam: foam / n });
      }
      return res;
    }

    // Records compute work for one frame. Returns the index of the slope texture holding the fresh result.
    encode(encoder, time, dt, timestampWrites) {
      const gpu = this.gpu;
      gpu.queue.writeBuffer(this.uBuf, 0, this.uniformData(time, dt));
      const pass = encoder.beginComputePass({ label: 'waves', timestampWrites: CV.tw('waves') || timestampWrites });
      if (this.dirty) {
        pass.setPipeline(this.pInit); pass.setBindGroup(0, this.bgInit); pass.dispatchWorkgroups(N / 8, N / 8, CASCADES);
        this.dirty = false;
      }
      pass.setPipeline(this.pEvolve); pass.setBindGroup(0, this.bgEvolve); pass.dispatchWorkgroups(N / 8, N / 8, CASCADES);
      pass.setPipeline(this.pRows); pass.setBindGroup(0, this.bgRows); pass.dispatchWorkgroups(N, CASCADES, 1);
      pass.setPipeline(this.pCols); pass.setBindGroup(0, this.bgCols); pass.dispatchWorkgroups(N, CASCADES, 1);
      const cur = this.frame & 1;
      pass.setPipeline(this.pAsm); pass.setBindGroup(0, this.bgAsm[cur]); pass.dispatchWorkgroups(N / 8, N / 8, CASCADES);
      this.mip.encode(pass, this.mipDisp); this.mip.encode(pass, this.mipSlope[cur]);
      pass.end();
      this.cur = cur; this.frame++;
      return cur;
    }
  };
  CV.Waves.N = N; CV.Waves.CASCADES = CASCADES;
})();
