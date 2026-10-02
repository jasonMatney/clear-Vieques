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
      tex(3), tex(4), tex(5), tex(6), tex(7, '2d-array'), tex(8, '2d-array'), tex(9), tex(10), tex(11), tex(12), tex(13), tex(14),
    ] });
  };

  CV.Renderer = class Renderer {
    constructor(gpu, terrain, sky, waves, ripples, caustics, mip, sceneLayout) {
      Object.assign(this, { gpu, terrain, sky, waves, ripples, caustics, mip, sceneLayout });
      this.sampleCount = 4; this.scale = 0.75; this.width = 0; this.height = 0;
      this.mesh = { seg: 576, rings: 260, r0: 0.5, growth: Math.pow(80000, 1 / 259) };
      this.debugView = 0;
      this.structs = new CV.Structures(gpu);
      this.timer = new CV.GpuTimer(gpu);
      this.taa = new CV.TAA(gpu);
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

      this.foliage = CV.makeFoliageTexture(gpu);        // tileable leaf-cluster structure for tree crowns (CPU-baked, mip-mapped)

      this.buildTerrainBindings();

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
      await this.structs.ready;
      this.postBGL = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
      this.upBGL = dev.createBindGroupLayout({ entries: [
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }] });
      this.pBuf = CV.buffer(gpu, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'postU');
      this.buildPipelines();
    }

    // (Re)creates every bind group that references the current terrain: scene bind groups (DEM, aux, canopy textures) and the mesh uniforms.
    buildTerrainBindings() {
      const dev = this.gpu.device, gpu = this.gpu;
      const make = (i) => dev.createBindGroup({ layout: this.sceneLayout, entries: [
        { binding: 0, resource: { buffer: this.gBuf } }, { binding: 1, resource: this.samLin }, { binding: 2, resource: this.samRep },
        { binding: 3, resource: this.terrain.demNear.createView() }, { binding: 4, resource: this.terrain.demFar.createView() },
        { binding: 5, resource: this.terrain.auxTex.createView() }, { binding: 6, resource: this.sky.tex.createView() },
        { binding: 7, resource: this.waves.dispView }, { binding: 8, resource: this.waves.slopeView[i] },
        { binding: 9, resource: this.ripples.out.createView() }, { binding: 10, resource: this.caustics.sceneView },
        { binding: 11, resource: this.noise.createView() },
        { binding: 12, resource: this.terrain.crownTex.createView() }, { binding: 13, resource: this.terrain.meanTex.createView() },
        { binding: 14, resource: this.foliage.createView() }] });
      this.sceneBG = [make(0), make(1)];
      // terrain mesh uniforms
      const mu = (g, isFar) => { const b = CV.buffer(gpu, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'meshU');
        gpu.queue.writeBuffer(b, 0, new Float32Array([g.x0, g.z0, g.dx, g.dz, g.nx, g.nz, isFar ? 1 : 0, 0])); return b; };
      const meshLayout = this.meshLayout || dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }] });
      this.meshLayout = meshLayout;
      this.meshBG = [this.terrain.meshNear.grid, this.terrain.meshFar.grid].map((g, i) => dev.createBindGroup({ layout: meshLayout, entries: [{ binding: 0, resource: { buffer: mu(g, i === 1) } }] }));
      this.structs.setSite(this.terrain);
    }

    // Site switch: adopt a new terrain (the previous one is destroyed by the caller).
    setTerrain(terrain) { this.terrain = terrain; this.buildTerrainBindings(); }

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
      // 1 m tiles near the camera (instanced; per-instance tile origins in a small uniform array)
      this.fineLayout = this.fineLayout || dev.createBindGroupLayout({ entries: [{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }] });
      if (!this.fineBuf) {
        this.fineBuf = CV.buffer(gpu, 32 * 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'fineTiles');
        this.fineBG = dev.createBindGroup({ layout: this.fineLayout, entries: [{ binding: 1, resource: { buffer: this.fineBuf } }] });
        this.fineData = new Float32Array(32 * 4);
      }
      this.pTerrainFine = dev.createRenderPipeline({ label: 'terrainFine', layout: dev.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.fineLayout] }),
        vertex: { module: mTerr, entryPoint: 'vs_terrain_fine' }, fragment: { module: mTerr, entryPoint: 'fs_terrain', targets: [{ format: hdr }] }, multisample: ms,
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: { ...depth, depthWriteEnabled: true } });
      this.pStruct = this.structs.pipeline(this.sceneLayout, this.sampleCount);
      this.pWater = dev.createRenderPipeline({ label: 'water', layout: lay1, vertex: { module: mWater, entryPoint: 'vs_water' },
        fragment: { module: mWater, entryPoint: 'fs_water', targets: [{ format: hdr, blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] }, multisample: ms,
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: { ...depth, depthWriteEnabled: true } });
      this.taa.pipeline(this.sampleCount);
      const postLay = dev.createPipelineLayout({ bindGroupLayouts: [this.postBGL] });
      this.pPost = dev.createRenderPipeline({ label: 'post', layout: postLay,
        vertex: { module: mPost, entryPoint: 'vs' }, fragment: { module: mPost, entryPoint: 'fs', targets: [{ format: this.gpu.format }] } });
      this.pGrade = dev.createRenderPipeline({ label: 'grade', layout: postLay,
        vertex: { module: mPost, entryPoint: 'vs' }, fragment: { module: mPost, entryPoint: 'fs_grade', targets: [{ format: 'rgba8unorm' }] } });
      this.pUp = dev.createRenderPipeline({ label: 'upscale', layout: dev.createPipelineLayout({ bindGroupLayouts: [this.upBGL] }),
        vertex: { module: mPost, entryPoint: 'vs' }, fragment: { module: mPost, entryPoint: 'fs_up', targets: [{ format: this.gpu.format }] } });
    }

    // w,h = internal render size in pixels; tw,th = size of the final image. When they differ the frame is graded into an LDR texture
    // at the internal size and upscaled edge-adaptively (see CV.wgsl.post); otherwise it is graded straight to the target.
    resize(w, h, tw, th) {
      w = Math.max(64, Math.floor(w)); h = Math.max(64, Math.floor(h));
      const up = w < Math.floor(tw) || h < Math.floor(th);
      if (w === this.width && h === this.height && this.hdr && up === this.upscale) return;
      const dev = this.gpu.device; this.width = w; this.height = h; this.upscale = up;
      for (const t of [this.hdr, this.hdrMS, this.depth, this.ldr]) if (t) t.destroy();
      const RA = GPUTextureUsage.RENDER_ATTACHMENT;
      this.hdr = dev.createTexture({ label: 'hdr', size: [w, h], format: 'rgba16float', usage: RA | GPUTextureUsage.TEXTURE_BINDING });
      this.hdrMS = this.sampleCount > 1 ? dev.createTexture({ label: 'hdrMS', size: [w, h], format: 'rgba16float', sampleCount: this.sampleCount, usage: RA }) : null;
      this.depth = dev.createTexture({ label: 'depth', size: [w, h], format: 'depth32float', sampleCount: this.sampleCount, usage: RA | GPUTextureUsage.TEXTURE_BINDING });
      this.ldr = up ? dev.createTexture({ label: 'ldr', size: [w, h], format: 'rgba8unorm', usage: RA | GPUTextureUsage.TEXTURE_BINDING }) : null;
      this.hdrView = this.hdr.createView(); this.hdrMSView = this.hdrMS && this.hdrMS.createView(); this.depthView = this.depth.createView();
      this.ldrView = this.ldr && this.ldr.createView();
      const postBG = (view) => dev.createBindGroup({ layout: this.postBGL, entries: [{ binding: 0, resource: view }, { binding: 1, resource: this.samPost }, { binding: 2, resource: { buffer: this.pBuf } }] });
      this.postBG = postBG(this.hdrView);
      this.taa.resize(w, h, this.hdrView, this.depthView);              // history pair; grading then reads whichever holds this frame's result
      this.postBGTaa = this.taa.views.map(postBG);
      this.upBG = up ? dev.createBindGroup({ layout: this.upBGL, entries: [{ binding: 2, resource: { buffer: this.pBuf } }, { binding: 3, resource: this.ldrView }] }) : null;
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
      // Night: the moon replaces the sun in every light path (direct light, sky LUT, glitter, shadows) with a phase-dependent gain K.
      const st = this.sky.sunTrans, night = !!s.night, K = night ? CV.Night.gain(s.moon) : 1, tn = night ? [1.0, 0.97, 0.90] : [1, 1, 1];
      G.set('sunE', Math.PI * st[0] * K * tn[0], Math.PI * st[1] * K * tn[1], Math.PI * st[2] * K * tn[2], s.exposure);
      const Ks = night ? K * CV.Night.SKY : 1;                        // scattered light is dimmed relative to direct moonlight (see CV.Night.SKY)
      G.set('skyE', this.sky.skyIrr[0] * Ks, this.sky.skyIrr[1] * Ks, this.sky.skyIrr[2] * Ks, Ks);   // .w scales the sky-LUT radiance
      G.set('horizon', this.sky.horizon[0] * Ks, this.sky.horizon[1] * Ks, this.sky.horizon[2] * Ks, s.airExt);
      const rel = night ? CV.Night.moonRel(s.moon) : 1;
      G.set('night', rel, night ? CV.Night.signedAlpha(s.moon) : 0, K, night ? CV.Night.starVis(s.moon) : 0);
      // bioluminescence: emitted light is fixed, but the eye/camera adapts, so it reads brighter as the moon thins ((FULL/K)^0.82, capped)
      const adapt = night ? Math.min(Math.pow(CV.Night.FULL / K, 0.82), 5) : 1;
      G.set('bio', night ? (s.biolum || 0) : 0, 1, adapt, 0);
      G.set('paddle', ...(s.paddle || [0, 0, 0, 0]));
      G.set('frame', ...(this.frameInfo || [0, 0, 0, 0]));
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
      G.set('crownA', tm.crown.x0, tm.crown.z0, tm.crown.texel, tm.crown.nx);
      G.set('crownB', tm.crown.nz, n.dx, tm.crown.maxH, 0);
      for (let i = 0; i < 8; i++) G.set('st' + i, ...this.structs.slots.subarray(i * 4, i * 4 + 4));
      G.set('ripple', ...this.ripples.uniforms());
      const cu = this.caustics.uniforms(); G.set('caustic', ...cu.a); G.set('causticB', ...cu.b);
      const mss = this.waves.mss(); G.set('shore', mss[0], mss[1], mss[2], 0);
      G.set('misc', s.quality || 1, s.night ? 1 : 0, s.moon || 0, s.matSet || 0);
      G.set('tint', this.debugView, s.cloudCover === undefined ? (this.cloudCover === undefined ? 0.36 : this.cloudCover) : s.cloudCover, this.opticsTest ? 1 : 0, this.noCrown ? 1 : 0);
      this.gpu.queue.writeBuffer(this.gBuf, 0, G.data);
    }

    // ------------------------------------------------------------------------------------------ frame
    // targetView: GPUTextureView of the final LDR image (canvas texture or an offscreen texture).
    frame(s, targetView, targetW, targetH) {
      const gpu = this.gpu, dev = gpu.device;
      this.resize(targetW * this.scale, targetH * this.scale, targetW, targetH);
      // dependent quantities for the sim / caustics
      const cam = s.cam;
      const sunVec = [Math.sin(s.sunAz * CV.D2R) * Math.cos(s.sunElev * CV.D2R), Math.sin(s.sunElev * CV.D2R), -Math.cos(s.sunAz * CV.D2R) * Math.cos(s.sunElev * CV.D2R)];
      this.caustics.place(s.causticCenter, s.meanDepth, sunVec);   // fixes the map window BEFORE the globals that describe it are written
      // TAA: offset the projection by this frame's sub-pixel jitter (the unjittered camera basis is what the TAA pass reprojects with)
      const taa = this.taa.enabled && !this.capturing;
      let jit = [0, 0];
      if (taa) {
        jit = this.taa.jitter(this.width, this.height);
        const J = CV.m4.identity(); J[12] = jit[0]; J[13] = jit[1];
        s.cam.viewProj = CV.m4.mul(J, s.cam.viewProj);
      }
      this.frameInfo = [taa ? this.taa.frame % 64 : 0, jit[0], jit[1], taa ? 1 : 0];
      this.updateGlobals(s);
      const enc = dev.createCommandEncoder({ label: 'frame' });
      const cur = this.waves.encode(enc, s.time, s.dt, this.timer.first());
      const focus = s.focus;
      this.ripples.encode(enc, s.dt, focus, this.terrain.seaLevel + s.seaOffset);
      // caustics need the scene bind group of the fresh wave textures
      this.caustics.sceneBG = this.sceneBG[cur];
      this.caustics.encode(enc);
      const ms = this.sampleCount > 1;
      const pass = enc.beginRenderPass({ label: 'scene',
        colorAttachments: [{ view: ms ? this.hdrMSView : this.hdrView, resolveTarget: ms ? this.hdrView : undefined, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: ms ? 'discard' : 'store' }],
        depthStencilAttachment: { view: this.depthView, depthLoadOp: 'clear', depthClearValue: 0, depthStoreOp: taa ? 'store' : 'discard' },
        timestampWrites: CV.prof ? CV.tw('scene') : this.timer.last() });   // end of the offscreen scene pass: excludes the swap-chain (vsync) wait of the post pass
      pass.setBindGroup(0, this.sceneBG[cur]);
      const skip = this.skip || {};
      if (!skip.sky) { pass.setPipeline(this.pSky); pass.draw(3); }
      if (!skip.terrain) {
        pass.setPipeline(this.pTerrain);
        // tiles that are off-screen, or lie wholly below the waterline (the water pass replaces them), are not drawn
        const planes = CV.frustumPlanes(s.cam.viewProj), seaY = this.terrain.seaLevel + s.seaOffset - 0.75;
        let drawn = 0;
        const far = this.terrain.meshFar, near = this.terrain.meshNear;
        pass.setBindGroup(1, this.meshBG[1]); pass.setIndexBuffer(far.buf, 'uint32');
        for (const t of far.tiles) { if (t.y1 < seaY || !CV.boxInFrustum(planes, t)) continue; pass.drawIndexed(t.count, 1, t.first); drawn++; }
        // near tiles within FINE_R of the camera are drawn at 1 m (instanced), the rest at 2 m
        const FINE_R = 140, cx = s.cam.pos[0], cz = s.cam.pos[2], fd = this.fineData; let nFine = 0;
        pass.setBindGroup(1, this.meshBG[0]); pass.setIndexBuffer(near.idx, 'uint32');
        for (const t of near.tiles) {
          if (t.y1 < seaY || !CV.boxInFrustum(planes, t)) continue;
          const dx = Math.max(t.x0 - cx, 0, cx - t.x1), dz = Math.max(t.z0 - cz, 0, cz - t.z1);
          if (nFine < 32 && Math.hypot(dx, dz) < FINE_R) { fd.set([t.x0, t.z0, near.fine.cell, 0], nFine * 4); nFine++; continue; }
          pass.drawIndexed(t.count, 1, 0, t.base); drawn++;
        }
        if (nFine > 0) {
          gpu.queue.writeBuffer(this.fineBuf, 0, fd, 0, nFine * 4);
          pass.setPipeline(this.pTerrainFine); pass.setBindGroup(1, this.fineBG); pass.setIndexBuffer(near.fine.idx, 'uint32');
          pass.drawIndexed(near.fine.count, nFine);
        }
        this.tilesDrawn = drawn; this.fineTilesDrawn = nFine;
      }
      if (!skip.terrain && this.structs.count > 0) {
        pass.setPipeline(this.pStruct); pass.setBindGroup(1, this.structs.bg); pass.setVertexBuffer(0, this.structs.vbuf);
        pass.draw(this.structs.vertexCount, this.structs.count);
      }
      if (!skip.water) { pass.setPipeline(this.pWater); pass.setIndexBuffer(this.waterIdx, 'uint32'); pass.drawIndexed(this.waterIdxCount); }
      pass.end();
      const postBG = taa ? this.postBGTaa[this.taa.encode(enc, s.cam, jit, 0.12)] : this.postBG;
      gpu.queue.writeBuffer(this.pBuf, 0, new Float32Array([s.exposureLin, s.vignette, (s.time * 60) % 1000, 1.12, this.debugView === 7 ? 1 : 0, this.width, this.height, s.night ? 1 : 0,
        taa ? 0.8 : 0, 0, 0, 0]));
      if (this.upscale) {
        const g = enc.beginRenderPass({ label: 'grade', timestampWrites: CV.tw('grade'), colorAttachments: [{ view: this.ldrView, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
        g.setPipeline(this.pGrade); g.setBindGroup(0, postBG); g.draw(3); g.end();
        const u = enc.beginRenderPass({ label: 'upscale', timestampWrites: CV.tw('upscale'), colorAttachments: [{ view: targetView, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
        u.setPipeline(this.pUp); u.setBindGroup(0, this.upBG); u.draw(3); u.end();
      } else {
        const post = enc.beginRenderPass({ label: 'post', timestampWrites: CV.tw('post'), colorAttachments: [{ view: targetView, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
        post.setPipeline(this.pPost); post.setBindGroup(0, postBG); post.draw(3); post.end();
      }
      if (CV.prof) CV.prof.finish(enc); else this.timer.resolveInto(enc);
      dev.queue.submit([enc.finish()]);
      this.timer.readback();
    }

    // Render one frame into an offscreen texture and return RGBA8 pixels.
    async capture(state, w, h) {
      const gpu = this.gpu, dev = gpu.device, fmt = gpu.format;
      const tex = dev.createTexture({ size: [w, h], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const savedScale = this.scale; this.scale = 1;
      this.capturing = true;                                          // a one-off frame at another size: no jitter, no history
      this.frame(state, tex.createView(), w, h);
      this.capturing = false;
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
