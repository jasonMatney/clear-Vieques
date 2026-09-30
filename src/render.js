// ClearVieques — renderer: shared scene bind group, HDR targets (+MSAA), sky / terrain / water passes, tone-map post, capture.
(function () {
  'use strict';
  const CV = window.CV;

  // ---- Jerlov-type water: attenuation K (1/m, RGB) and deep-water reflectance. Slider 0..1 walks I -> 9C.
  const JERLOV = [
    ['I',   [0.360, 0.058, 0.021], [0.00035, 0.0075, 0.0200]],
    ['IA',  [0.364, 0.064, 0.027], [0.00040, 0.0082, 0.0195]],
    ['IB',  [0.370, 0.072, 0.035], [0.00050, 0.0092, 0.0190]],
    ['II',  [0.385, 0.090, 0.054], [0.00070, 0.0110, 0.0180]],
    ['III', [0.420, 0.125, 0.090], [0.00110, 0.0140, 0.0170]],
    ['1C',  [0.470, 0.185, 0.170], [0.00200, 0.0180, 0.0150]],
    ['3C',  [0.560, 0.290, 0.300], [0.00380, 0.0230, 0.0130]],
    ['5C',  [0.720, 0.430, 0.450], [0.00600, 0.0280, 0.0120]],
    ['7C',  [0.960, 0.630, 0.640], [0.00900, 0.0320, 0.0110]],
    ['9C',  [1.300, 0.900, 0.880], [0.01200, 0.0350, 0.0100]],
  ];
  CV.jerlov = function (t) {
    const x = CV.clamp(t, 0, 1) * (JERLOV.length - 1), i = Math.min(JERLOV.length - 2, Math.floor(x)), f = x - i;
    const a = JERLOV[i], b = JERLOV[i + 1];
    const mixv = (p, q) => p.map((v, k) => v + (q[k] - v) * f);
    return { name: f < 0.5 ? a[0] : b[0], K: mixv(a[1], b[1]), R: mixv(a[2], b[2]) };
  };

  CV.makeSceneLayout = function (gpu) {
    const all = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    const tex = (binding, viewDimension = '2d') => ({ binding, visibility: all, texture: { sampleType: 'float', viewDimension } });
    return gpu.device.createBindGroupLayout({ label: 'scene', entries: [
      { binding: 0, visibility: all, buffer: { type: 'uniform' } },
      { binding: 1, visibility: all, sampler: { type: 'filtering' } },
      { binding: 2, visibility: all, sampler: { type: 'filtering' } },
      tex(3), tex(4), tex(5), tex(6), tex(7, '2d-array'), tex(8, '2d-array'), tex(9), tex(10), tex(11),
    ] });
  };

  CV.Renderer = class Renderer {
    constructor(gpu, terrain, sky, waves, ripples, caustics, mip, sceneLayout) {
      Object.assign(this, { gpu, terrain, sky, waves, ripples, caustics, mip, sceneLayout });
      this.sampleCount = 4; this.scale = 0.75; this.width = 0; this.height = 0;
      this.mesh = { seg: 576, rings: 260, r0: 0.5, growth: Math.pow(80000, 1 / 259) };
      this.debugView = 0;
      this.timer = new CV.GpuTimer(gpu);
      this.ready = this.init();
    }

    async init() {
      const gpu = this.gpu, dev = gpu.device;
      this.sceneLayout = this.sceneLayout || CV.makeSceneLayout(gpu);
      this.samLin = dev.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
      this.samRep = dev.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 16 });
      this.samPost = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });
      this.gBuf = CV.buffer(gpu, CV.globals.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'globals');

      // random RGBA8 noise (value-noise source)
      const rnd = CV.mulberry32(20240915), nb = new Uint8Array(256 * 256 * 4);
      for (let i = 0; i < nb.length; i++) nb[i] = Math.floor(rnd() * 256);
      this.noise = dev.createTexture({ label: 'noise', size: [256, 256], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      gpu.queue.writeTexture({ texture: this.noise }, nb, { bytesPerRow: 1024 }, [256, 256]);

      // shared scene bind groups (one per wave-slope ping-pong texture)
      const make = (i) => dev.createBindGroup({ layout: this.sceneLayout, entries: [
        { binding: 0, resource: { buffer: this.gBuf } }, { binding: 1, resource: this.samLin }, { binding: 2, resource: this.samRep },
        { binding: 3, resource: this.terrain.demNear.createView() }, { binding: 4, resource: this.terrain.demFar.createView() },
        { binding: 5, resource: this.terrain.auxTex.createView() }, { binding: 6, resource: this.sky.tex.createView() },
        { binding: 7, resource: this.waves.dispView }, { binding: 8, resource: this.waves.slopeView[i] },
        { binding: 9, resource: this.ripples.out.createView() }, { binding: 10, resource: this.caustics.sceneView },
        { binding: 11, resource: this.noise.createView() }] });
      this.sceneBG = [make(0), make(1)];

      // terrain mesh uniforms
      const mu = (g, isFar) => { const b = CV.buffer(gpu, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'meshU');
        gpu.queue.writeBuffer(b, 0, new Float32Array([g.x0, g.z0, g.dx, g.dz, g.nx, g.nz, isFar ? 1 : 0, 0])); return b; };
      const meshLayout = dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }] });
      this.meshBG = [this.terrain.near, this.terrain.far].map((g, i) => dev.createBindGroup({ layout: meshLayout, entries: [{ binding: 0, resource: { buffer: mu(g, i === 1) } }] }));

      // water mesh indices
      const { seg, rings } = this.mesh, idx = new Uint32Array((rings - 1) * seg * 6); let k = 0;
      for (let r = 0; r < rings - 1; r++) for (let s = 0; s < seg; s++) {
        const a = r * seg + s, b = r * seg + (s + 1) % seg, c = (r + 1) * seg + s, d = (r + 1) * seg + (s + 1) % seg;
        idx[k++] = a; idx[k++] = c; idx[k++] = b; idx[k++] = b; idx[k++] = c; idx[k++] = d;
      }
      this.waterIdx = dev.createBuffer({ label: 'waterIdx', size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      gpu.queue.writeBuffer(this.waterIdx, 0, idx); this.waterIdxCount = idx.length;

      // shaders
      const prelude = CV.wgsl.prelude();
      const [mSky, mTerr, mWater, mPost] = await Promise.all([
        CV.shader(gpu, 'sky', prelude + CV.wgsl.sky), CV.shader(gpu, 'terrain', prelude + CV.wgsl.terrain()),
        CV.shader(gpu, 'water', prelude + CV.wgsl.water(this.mesh)), CV.shader(gpu, 'post', CV.wgsl.post)]);
      this.modules = { mSky, mTerr, mWater, mPost };
      this.meshLayout = meshLayout;
      this.postBGL = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
      this.pBuf = CV.buffer(gpu, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'postU');
      this.buildPipelines();
    }

    buildPipelines() {
      const gpu = this.gpu, dev = gpu.device, ms = { count: this.sampleCount }, { mSky, mTerr, mWater, mPost } = this.modules;
      const hdr = 'rgba16float', depth = { format: 'depth32float', depthCompare: 'greater' };
      const lay1 = dev.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout] });
      const lay2 = dev.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.meshLayout] });
      this.pSky = dev.createRenderPipeline({ label: 'sky', layout: lay1, vertex: { module: mSky, entryPoint: 'vs_sky' },
        fragment: { module: mSky, entryPoint: 'fs_sky', targets: [{ format: hdr }] }, multisample: ms,
        depthStencil: { format: 'depth32float', depthCompare: 'always', depthWriteEnabled: false } });
      this.pTerrain = dev.createRenderPipeline({ label: 'terrain', layout: lay2, vertex: { module: mTerr, entryPoint: 'vs_terrain' },
        fragment: { module: mTerr, entryPoint: 'fs_terrain', targets: [{ format: hdr }] }, multisample: ms,
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: { ...depth, depthWriteEnabled: true } });
      this.pWater = dev.createRenderPipeline({ label: 'water', layout: lay1, vertex: { module: mWater, entryPoint: 'vs_water' },
        fragment: { module: mWater, entryPoint: 'fs_water', targets: [{ format: hdr, blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] }, multisample: ms,
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: { ...depth, depthWriteEnabled: true } });
      this.pPost = dev.createRenderPipeline({ label: 'post', layout: dev.createPipelineLayout({ bindGroupLayouts: [this.postBGL] }),
        vertex: { module: mPost, entryPoint: 'vs' }, fragment: { module: mPost, entryPoint: 'fs', targets: [{ format: this.gpu.format }] } });
    }

    resize(w, h) { // w,h = internal render size in pixels
      w = Math.max(64, Math.floor(w)); h = Math.max(64, Math.floor(h));
      if (w === this.width && h === this.height && this.hdr) return;
      const dev = this.gpu.device; this.width = w; this.height = h;
      for (const t of [this.hdr, this.hdrMS, this.depth]) if (t) t.destroy();
      const RA = GPUTextureUsage.RENDER_ATTACHMENT;
      this.hdr = dev.createTexture({ label: 'hdr', size: [w, h], format: 'rgba16float', usage: RA | GPUTextureUsage.TEXTURE_BINDING });
      this.hdrMS = this.sampleCount > 1 ? dev.createTexture({ label: 'hdrMS', size: [w, h], format: 'rgba16float', sampleCount: this.sampleCount, usage: RA }) : null;
      this.depth = dev.createTexture({ label: 'depth', size: [w, h], format: 'depth32float', sampleCount: this.sampleCount, usage: RA });
      this.hdrView = this.hdr.createView(); this.hdrMSView = this.hdrMS && this.hdrMS.createView(); this.depthView = this.depth.createView();
      this.postBG = dev.createBindGroup({ layout: this.postBGL, entries: [{ binding: 0, resource: this.hdrView }, { binding: 1, resource: this.samPost }, { binding: 2, resource: { buffer: this.pBuf } }] });
    }

    setSamples(n) { if (n === this.sampleCount) return; this.sampleCount = n; this.width = 0; this.buildPipelines(); }

    // ------------------------------------------------------------------------------------------ per-frame uniforms
    updateGlobals(s) {
      const G = CV.globals, cam = s.cam, sea = this.terrain.seaLevel + s.seaOffset, tm = this.terrain;
      G.setMat('viewProj', cam.viewProj);
      G.set('camPos', cam.pos[0], cam.pos[1], cam.pos[2], s.time);
      G.set('camRight', cam.right[0], cam.right[1], cam.right[2], cam.tanX);
      G.set('camUp', cam.up[0], cam.up[1], cam.up[2], cam.tanY);
      G.set('camFwd', cam.fwd[0], cam.fwd[1], cam.fwd[2], 2 * cam.tanY / this.height);
      G.set('screen', this.width, this.height, 1 / this.width, 1 / this.height);
      const el = s.sunElev * CV.D2R, az = s.sunAz * CV.D2R;
      G.set('sunDir', Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el), el);
      const st = this.sky.sunTrans;
      G.set('sunE', Math.PI * st[0], Math.PI * st[1], Math.PI * st[2], s.exposure);
      G.set('skyE', ...this.sky.skyIrr, 0);
      G.set('horizon', ...this.sky.horizon, s.airExt);
      const j = CV.jerlov(s.turbidity);
      G.set('kAbs', j.K[0], j.K[1], j.K[2], sea);
      G.set('rDeep', j.R[0] * s.deepGain, j.R[1] * s.deepGain, j.R[2] * s.deepGain, s.turbidity);
      const w = this.waves, p = w.params;
      G.set('waveA', p.choppy, p.foamGain, p.wind, p.energy);
      const rot = [0, 37 * CV.D2R, -71 * CV.D2R];
      [0, 1, 2].forEach(c => G.set('casc' + c, w.L[c], Math.cos(rot[c]), Math.sin(rot[c]), 1));
      const n = tm.near, f = tm.far;
      G.set('demNear', n.x0, n.z0, n.nx * n.dx, n.nz * n.dz);
      G.set('demFar', f.x0, f.z0, f.nx * f.dx, f.nz * f.dz);
      G.set('demInfo', tm.blend, 0, 0, 0);
      G.set('ripple', ...this.ripples.uniforms());
      const cu = this.caustics.uniforms(); G.set('caustic', ...cu.a); G.set('causticB', ...cu.b);
      const mss = this.waves.mss(); G.set('shore', mss[0], mss[1], mss[2], 0);
      G.set('misc', s.quality || 1, s.night ? 1 : 0, s.moon || 0, 0);
      G.set('tint', this.debugView, s.cloudCover === undefined ? 0.36 : s.cloudCover, this.opticsTest ? 1 : 0, 0);
      this.gpu.queue.writeBuffer(this.gBuf, 0, G.data);
    }

    // ------------------------------------------------------------------------------------------ frame
    // targetView: GPUTextureView of the final LDR image (canvas texture or an offscreen texture).
    frame(s, targetView, targetW, targetH) {
      const gpu = this.gpu, dev = gpu.device;
      this.resize(targetW * this.scale, targetH * this.scale);
      // dependent quantities for the sim / caustics
      const cam = s.cam;
      this.updateGlobals(s);
      const enc = dev.createCommandEncoder({ label: 'frame' });
      const cur = this.waves.encode(enc, s.time, s.dt, this.timer.first());
      const focus = s.focus;
      this.ripples.encode(enc, s.dt, focus, this.terrain.seaLevel + s.seaOffset);
      // caustics need the scene bind group of the fresh wave textures
      this.caustics.sceneBG = this.sceneBG[cur];
      this.caustics.encode(enc, s.causticCenter, s.meanDepth, [Math.sin(s.sunAz * CV.D2R) * Math.cos(s.sunElev * CV.D2R), Math.sin(s.sunElev * CV.D2R), -Math.cos(s.sunAz * CV.D2R) * Math.cos(s.sunElev * CV.D2R)]);
      const ms = this.sampleCount > 1;
      const pass = enc.beginRenderPass({ label: 'scene',
        colorAttachments: [{ view: ms ? this.hdrMSView : this.hdrView, resolveTarget: ms ? this.hdrView : undefined, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: ms ? 'discard' : 'store' }],
        depthStencilAttachment: { view: this.depthView, depthLoadOp: 'clear', depthClearValue: 0, depthStoreOp: 'discard' },
        timestampWrites: CV.prof ? CV.tw('scene') : this.timer.last() });   // end of the offscreen scene pass: excludes the swap-chain (vsync) wait of the post pass
      pass.setBindGroup(0, this.sceneBG[cur]);
      const skip = this.skip || {};
      if (!skip.sky) { pass.setPipeline(this.pSky); pass.draw(3); }
      if (!skip.terrain) {
        pass.setPipeline(this.pTerrain);
        pass.setBindGroup(1, this.meshBG[1]); pass.setIndexBuffer(this.terrain.meshFar.buf, 'uint32'); pass.drawIndexed(this.terrain.meshFar.count);
        pass.setBindGroup(1, this.meshBG[0]); pass.setIndexBuffer(this.terrain.meshNear.buf, 'uint32'); pass.drawIndexed(this.terrain.meshNear.count);
      }
      if (!skip.water) { pass.setPipeline(this.pWater); pass.setIndexBuffer(this.waterIdx, 'uint32'); pass.drawIndexed(this.waterIdxCount); }
      pass.end();
      gpu.queue.writeBuffer(this.pBuf, 0, new Float32Array([s.exposureLin, s.vignette, (s.time * 60) % 1000, 1.12]));
      const post = enc.beginRenderPass({ label: 'post', timestampWrites: CV.tw('post'), colorAttachments: [{ view: targetView, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
      post.setPipeline(this.pPost); post.setBindGroup(0, this.postBG); post.draw(3); post.end();
      if (CV.prof) CV.prof.finish(enc); else this.timer.resolveInto(enc);
      dev.queue.submit([enc.finish()]);
      this.timer.readback();
    }

    // Render one frame into an offscreen texture and return RGBA8 pixels.
    async capture(state, w, h) {
      const gpu = this.gpu, dev = gpu.device, fmt = gpu.format;
      const tex = dev.createTexture({ size: [w, h], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const savedScale = this.scale; this.scale = 1;
      if (this.waves.frame < 3) for (let i = 0; i < 3; i++) this.frame(state, tex.createView(), w, h); // warm-up: the very first frame has no caustic history
      this.frame(state, tex.createView(), w, h);
      const bpr = Math.ceil(w * 4 / 256) * 256;
      const buf = dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = dev.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]); dev.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(buf.getMappedRange()), out = new Uint8ClampedArray(w * h * 4), bgra = fmt.startsWith('bgra');
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * bpr + x * 4, o = (y * w + x) * 4;
        out[o] = src[i + (bgra ? 2 : 0)]; out[o + 1] = src[i + 1]; out[o + 2] = src[i + (bgra ? 0 : 2)]; out[o + 3] = 255;
      }
      buf.unmap(); buf.destroy(); tex.destroy(); this.scale = savedScale;
      return { w, h, data: out };
    }
  };
})();
