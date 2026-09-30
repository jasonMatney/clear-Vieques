// ClearVieques — caustics projected onto the real sampled seafloor.
// Pass 1 (compute): a grid of photons starts on the wavy surface, is refracted toward the sun-lit seabed and ray-marched to the DEM.
// Pass 2 (raster):  each photon triangle is splatted (additively) at its seabed footprint with intensity = source area / footprint area.
// The resulting map (mean ~1, mip-mapped) modulates the direct sunlight on the bed in the water shader.
(function () {
  'use strict';
  const CV = window.CV;

  const photonCode = () => CV.wgsl.prelude() + /* wgsl */`
struct PU { a : vec4<f32>, b : vec4<f32> };   // a: photon grid origin x, z, spacing, count (cells per side)
@group(1) @binding(0) var<uniform> PUn : PU;
@group(1) @binding(1) var photonOut : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn photons(@builtin(global_invocation_id) id : vec3<u32>) {
  let np = u32(PUn.a.w) + 1u;
  if (id.x >= np || id.y >= np) { return; }
  let s = PUn.a.xy + vec2<f32>(id.xy) * PUn.a.z;
  let sea = seaLevel();
  let d = sea - heightAt(s);
  if (d < 0.04) { textureStore(photonOut, vec2<i32>(id.xy), vec4<f32>(0.0)); return; }
  let ws = waveSlopeLevel(s, d, PUn.a.z);
  let n = normalize(vec3<f32>(-ws.x, 1.0, -ws.y));
  let L = G.sunDir.xyz;
  let ndl = dot(n, L);
  if (ndl <= 0.02) { textureStore(photonOut, vec2<i32>(id.xy), vec4<f32>(0.0)); return; }
  let rd = refract(-L, n, 1.0 / N_WATER);
  let P0 = vec3<f32>(s.x, sea + ws.z, s.y);
  let tEnd = clamp((d + 12.0) / max(-rd.y, 0.15), 3.0, 60.0);
  let hit = marchBed(P0, rd, tEnd);
  // flux relative to a flat surface (Fresnel transmittance and projected-area change)
  let flux = (ndl / (max(n.y, 0.05) * L.y)) * (1.0 - fresnelAir(ndl)) / max(1.0 - fresnelAir(L.y), 0.05);
  textureStore(photonOut, vec2<i32>(id.xy), vec4<f32>(hit.x, hit.z, hit.y, flux));
}
`;

  const splatCode = /* wgsl */`
struct CU { a : vec4<f32>, b : vec4<f32> };   // a: spacing, count, map origin x, z ; b: map size (m), max intensity
@group(0) @binding(0) var<uniform> C : CU;
@group(0) @binding(1) var photons : texture_2d<f32>;
struct COut { @builtin(position) pos : vec4<f32>, @location(0) inten : f32 };

@vertex fn vs(@builtin(vertex_index) vid : u32) -> COut {
  let tri = vid / 3u; let corner = vid % 3u;
  let np = u32(C.a.y);
  let cell = tri / 2u; let half = tri & 1u;
  let ci = cell % np; let cj = cell / np;
  var ox = array<u32, 6>(0u, 1u, 0u, 1u, 1u, 0u);
  var oy = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
  let base = half * 3u;
  let c0 = vec2<i32>(i32(ci + ox[base]), i32(cj + oy[base]));
  let c1 = vec2<i32>(i32(ci + ox[base + 1u]), i32(cj + oy[base + 1u]));
  let c2 = vec2<i32>(i32(ci + ox[base + 2u]), i32(cj + oy[base + 2u]));
  let a = textureLoad(photons, c0, 0); let b = textureLoad(photons, c1, 0); let c = textureLoad(photons, c2, 0);
  var o : COut;
  if (a.w <= 0.0 || b.w <= 0.0 || c.w <= 0.0) { o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0); o.inten = 0.0; return o; }
  let e1 = b.xy - a.xy; let e2 = c.xy - a.xy;
  let area = 0.5 * abs(e1.x * e2.y - e1.y * e2.x);
  let src = 0.5 * C.a.x * C.a.x;
  o.inten = min((a.w + b.w + c.w) * (1.0 / 3.0) * src / max(area, src * 0.02), C.b.y);
  let mine = select(select(a, b, corner == 1u), c, corner == 2u);
  let uv = (mine.xy - C.a.zw) / C.b.x;
  o.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  return o;
}
@fragment fn fs(in : COut) -> @location(0) vec4<f32> { return vec4<f32>(in.inten, 0.0, 0.0, 1.0); }
`;

  CV.Caustics = class Caustics {
    constructor(gpu, sceneLayout, renderer) {
      this.gpu = gpu; this.sceneLayout = sceneLayout;
      this.M = 2048;        // splat resolution (mip 1 is the sampled base -> 4 coverage samples per texel)
      this.NPMAX = 1120;    // photon texture is sized for the highest tier
      this.NP = 840;        // photon cells per side (quality tier)
      this.size = 28;       // map window (m)
      this.overscan = 1.3;  // photon window / map window
      this.origin = [0, 0]; this.enabled = true; this.intensity = 1.0;
      this.ready = this.init();
    }
    async init() {
      const gpu = this.gpu, dev = gpu.device, M = this.M, NP = this.NPMAX;
      this.tex = dev.createTexture({ label: 'causticMap', size: [M, M, 1], format: 'rgba16float', mipLevelCount: CV.mipCount(M, M),
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
      this.sceneView = this.tex.createView({ dimension: '2d' });
      this.rtView = this.tex.createView({ dimension: '2d', baseMipLevel: 0, mipLevelCount: 1 });
      this.photonTex = dev.createTexture({ label: 'photons', size: [NP + 1, NP + 1], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
      const pLayout = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } }] });
      this.uPhoton = CV.buffer(gpu, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'photonU');
      this.bgPhoton = dev.createBindGroup({ layout: pLayout, entries: [{ binding: 0, resource: { buffer: this.uPhoton } }, { binding: 1, resource: this.photonTex.createView() }] });
      const mod = await CV.shader(gpu, 'photons', photonCode());
      this.pPhoton = dev.createComputePipeline({ label: 'photons', layout: dev.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, pLayout] }), compute: { module: mod, entryPoint: 'photons' } });

      const sLayout = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } }] });
      this.uSplat = CV.buffer(gpu, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'splatU');
      this.bgSplat = dev.createBindGroup({ layout: sLayout, entries: [{ binding: 0, resource: { buffer: this.uSplat } }, { binding: 1, resource: this.photonTex.createView() }] });
      const sm = await CV.shader(gpu, 'causticSplat', splatCode);
      this.pSplat = dev.createRenderPipeline({ label: 'causticSplat', layout: dev.createPipelineLayout({ bindGroupLayouts: [sLayout] }),
        vertex: { module: sm, entryPoint: 'vs' },
        fragment: { module: sm, entryPoint: 'fs', targets: [{ format: 'rgba16float', blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' } });
      await this.mip?.ready;
      this.mipPasses = null;
    }
    attachMip(mip) { this.mip = mip; this.mipPasses = mip.makePasses(this.tex, this.M, this.M, 1); }

    // Place the map window and record the caustic passes. sunDir = unit vector toward the sun. focus = [x, z] centre of interest.
    encode(encoder, focus, meanDepth, sunDir) {
      if (!this.enabled || !this.mipPasses) return;
      const gpu = this.gpu, W = this.size, M = this.M, NP = this.NP;
      const texel = W / M, snapMap = texel * 16;
      const mc = [Math.round(focus[0] / snapMap) * snapMap, Math.round(focus[1] / snapMap) * snapMap];
      this.origin = mc;
      // horizontal shift of a refracted sun ray over the mean depth: photons must start up-sun of the map window
      const cosI = sunDir[1], sinI = Math.sqrt(Math.max(1 - cosI * cosI, 0)), sinT = sinI / 1.333, cosT = Math.sqrt(1 - sinT * sinT);
      const hl = Math.hypot(sunDir[0], sunDir[2]) || 1;
      const travel = [-sunDir[0] / hl * sinT / cosT * meanDepth, -sunDir[2] / hl * sinT / cosT * meanDepth];
      const PW = W * this.overscan, spacing = PW / NP;
      const pc = [mc[0] - travel[0], mc[1] - travel[1]];
      const po = [Math.round((pc[0] - PW / 2) / spacing) * spacing, Math.round((pc[1] - PW / 2) / spacing) * spacing];
      const mapOrigin = [mc[0] - W / 2, mc[1] - W / 2];
      gpu.queue.writeBuffer(this.uPhoton, 0, new Float32Array([po[0], po[1], spacing, NP, 0, 0, 0, 0]));
      gpu.queue.writeBuffer(this.uSplat, 0, new Float32Array([spacing, NP, mapOrigin[0], mapOrigin[1], W, 40, 0, 0]));
      // 1) photons (needs the scene bind group: waves, DEM)
      const cp = encoder.beginComputePass({ label: 'photons', timestampWrites: CV.tw('photons') });
      cp.setPipeline(this.pPhoton); cp.setBindGroup(0, this.sceneBG); cp.setBindGroup(1, this.bgPhoton);
      cp.dispatchWorkgroups(Math.ceil((NP + 1) / 8), Math.ceil((NP + 1) / 8), 1);
      cp.end();
      // 2) splat
      const rp = encoder.beginRenderPass({ label: 'causticSplat', timestampWrites: CV.tw('splat'), colorAttachments: [{ view: this.rtView, loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }] });
      rp.setPipeline(this.pSplat); rp.setBindGroup(0, this.bgSplat); rp.draw(NP * NP * 2 * 3);
      rp.end();
      // 3) mips
      const mp = encoder.beginComputePass({ label: 'causticMips', timestampWrites: CV.tw('causticMips') });
      this.mip.encode(mp, this.mipPasses);
      mp.end();
    }
    uniforms() { // caustic: origin x, z, size, intensity ; causticB: texel, max lod, fade start, fade end
      return { a: [this.origin[0], this.origin[1], this.size, this.enabled ? this.intensity : 0], b: [2 * this.size / this.M, Math.log2(this.M) - 3, 20, 34] };
    }
  };
})();

