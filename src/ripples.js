// ClearVieques — interactive ripples: a GPU wave-equation heightfield on a camera-following toroidal window.
// Wave speed follows the real bathymetry (c = sqrt(g d), capped), so ripples refract into and die on the beach.
(function () {
  'use strict';
  const CV = window.CV;
  const NR = 512, MAX_IMP = 8;

  const stepCode = (fmt) => /* wgsl */`
struct RU {
  win : vec4<f32>,     // new origin cell (x, z), old origin cell (x, z) — integer cell coordinates of the window corner
  p : vec4<f32>,       // cell size (m), dt (s), damping (1/s), sea level (m)
  q : vec4<f32>,       // impulse count, c max (m/s), agitation decay (1/s), unused
  imp : array<vec4<f32>, ${MAX_IMP}>,   // x, z, radius, strength (m)
};
@group(0) @binding(0) var<uniform> U : RU;
@group(0) @binding(1) var prevTex : texture_2d<f32>;
@group(0) @binding(2) var nextTex : texture_storage_2d<${fmt}, write>;
@group(0) @binding(3) var demNearTex : texture_2d<f32>;
@group(0) @binding(4) var demFarTex : texture_2d<f32>;
@group(0) @binding(5) var samLin : sampler;
@group(0) @binding(6) var<uniform> GD : array<vec4<f32>, 3>;   // demNear rect, demFar rect, (blend width, ...)

const NR : i32 = ${NR};
fn pm(a : i32, n : i32) -> i32 { return ((a % n) + n) % n; }
fn nearW(p : vec2<f32>) -> f32 {
  let o = GD[0].xy; let s = GD[0].zw;
  let d = min(min(p.x - o.x, o.x + s.x - p.x), min(p.y - o.y, o.y + s.y - p.y));
  return smoothstep(0.0, GD[2].x, d);
}
fn demH(p : vec2<f32>) -> f32 {
  var h = textureSampleLevel(demFarTex, samLin, (p - GD[1].xy) / GD[1].zw, 0.0).r;
  let w = nearW(p);
  if (w > 0.0) { h = mix(h, textureSampleLevel(demNearTex, samLin, (p - GD[0].xy) / GD[0].zw, 0.0).r, w); }
  return h;
}
// world cell represented by texel 'idx' for a window whose lower-left cell is 'org'
fn cellOf(idx : vec2<i32>, org : vec2<i32>) -> vec2<i32> { return org + vec2<i32>(pm(idx.x - org.x, NR), pm(idx.y - org.y, NR)); }

fn nbr(idx : vec2<i32>, off : vec2<i32>, cn : vec2<i32>, orgN : vec2<i32>, orgO : vec2<i32>) -> f32 {
  let ni = vec2<i32>(pm(idx.x + off.x, NR), pm(idx.y + off.y, NR));
  let cNew = cellOf(ni, orgN);
  if (cNew.x != cn.x + off.x || cNew.y != cn.y + off.y) { return 0.0; }        // across the window seam
  let cOld = cellOf(ni, orgO);
  if (cNew.x != cOld.x || cNew.y != cOld.y) { return 0.0; }                     // stale texel (cell just entered the window)
  return textureLoad(prevTex, ni, 0).x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(NR) || id.y >= u32(NR)) { return; }
  let idx = vec2<i32>(id.xy);
  let orgN = vec2<i32>(i32(U.win.x), i32(U.win.y));
  let orgO = vec2<i32>(i32(U.win.z), i32(U.win.w));
  let cn = cellOf(idx, orgN); let co = cellOf(idx, orgO);
  let cs = U.p.x;
  let pos = (vec2<f32>(cn) + 0.5) * cs;
  var st = textureLoad(prevTex, idx, 0);
  if (cn.x != co.x || cn.y != co.y) { st = vec4<f32>(0.0); }
  let depth = U.p.w - demH(pos);
  if (depth < 0.015) { textureStore(nextTex, idx, vec4<f32>(0.0)); return; }    // land: no water here
  var h = st.x; var v = st.y; var ag = st.z;
  let hl = nbr(idx, vec2<i32>(-1, 0), cn, orgN, orgO); let hr = nbr(idx, vec2<i32>(1, 0), cn, orgN, orgO);
  let hd = nbr(idx, vec2<i32>(0, -1), cn, orgN, orgO); let hu = nbr(idx, vec2<i32>(0, 1), cn, orgN, orgO);
  let lap = (hl + hr + hd + hu - 4.0 * h) / (cs * cs);
  let c2 = min(9.81 * max(depth, 0.02), U.q.y * U.q.y);
  let dt = U.p.y;
  v += c2 * lap * dt;
  let damp = U.p.z * (1.0 + 6.0 * (1.0 - smoothstep(0.02, 0.35, depth)));      // swash dissipation near the shoreline
  v *= exp(-damp * dt);
  h += v * dt;
  // absorbing border (in world-cell space, relative to the window)
  let e = max(abs(f32(cn.x - orgN.x) - 255.5), abs(f32(cn.y - orgN.y) - 255.5)) / 256.0;
  let edge = 1.0 - smoothstep(0.80, 0.98, e);
  let n = i32(U.q.x);
  for (var k = 0; k < ${MAX_IMP}; k++) {
    if (k >= n) { break; }
    let im = U.imp[k];
    let d2 = dot(pos - im.xy, pos - im.xy);
    let s2 = im.z * im.z;
    let g = exp(-d2 / (2.0 * s2));
    h += im.w * g * (1.0 - 0.25 * d2 / s2);
    ag = max(ag, min(abs(im.w) * 10.0, 1.0) * g);
  }
  ag = max(ag * exp(-U.q.z * dt), min(abs(v) * 1.2, 1.0));
  textureStore(nextTex, idx, vec4<f32>(h * edge, v * edge, ag * edge, 0.0));
}
`;

  // Output pass: (h, dh/dx, dh/dz, agitation) for rendering.
  const outCode = (fmt) => /* wgsl */`
@group(0) @binding(0) var<uniform> U : vec4<f32>;   // cell size in .x
@group(0) @binding(1) var stateTex : texture_2d<f32>;
@group(0) @binding(2) var outTex : texture_storage_2d<${fmt}, write>;
const NR : i32 = ${NR};
fn pm(a : i32, n : i32) -> i32 { return ((a % n) + n) % n; }
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(NR) || id.y >= u32(NR)) { return; }
  let idx = vec2<i32>(id.xy);
  let s = textureLoad(stateTex, idx, 0);
  let hl = textureLoad(stateTex, vec2<i32>(pm(idx.x - 1, NR), idx.y), 0).x;
  let hr = textureLoad(stateTex, vec2<i32>(pm(idx.x + 1, NR), idx.y), 0).x;
  let hd = textureLoad(stateTex, vec2<i32>(idx.x, pm(idx.y - 1, NR)), 0).x;
  let hu = textureLoad(stateTex, vec2<i32>(idx.x, pm(idx.y + 1, NR)), 0).x;
  textureStore(outTex, idx, vec4<f32>(s.x, (hr - hl) / (2.0 * U.x), (hu - hd) / (2.0 * U.x), s.z));
}
`;

  CV.Ripples = class Ripples {
    constructor(gpu, terrain) {
      this.gpu = gpu; this.terrain = terrain; this.NR = NR; this.cell = 0.08;  this.size = NR * this.cell;
      this.center = [0, 0]; this.pending = []; this.enabled = true; this.forceClear = false; this.oldOrigin = null;
      this.params = { cMax: 1.5, damping: 0.42, agitDecay: 1.3, gain: 2.4 };
      this.cur = 0;
      this.ready = this.init();
    }
    async init() {
      const gpu = this.gpu, dev = gpu.device, fmt = gpu.fmtSim, t = this.terrain;
      const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
      const mk = (label) => dev.createTexture({ label, size: [NR, NR], format: fmt, usage });
      this.state = [mk('rippleA'), mk('rippleB')]; this.out = mk('rippleOut');
      const [m1, m2] = await Promise.all([CV.shader(gpu, 'rippleStep', stepCode(fmt)), CV.shader(gpu, 'rippleOut', outCode(fmt))]);
      this.pStep = dev.createComputePipeline({ label: 'rippleStep', layout: 'auto', compute: { module: m1, entryPoint: 'main' } });
      this.pOut = dev.createComputePipeline({ label: 'rippleOut', layout: 'auto', compute: { module: m2, entryPoint: 'main' } });
      const ub = () => CV.buffer(gpu, (3 + MAX_IMP) * 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'rippleU');
      this.uBuf = [ub(), ub()]; // one per sub-step (queue.writeBuffer lands before the command buffer runs)
      this.gdBuf = CV.buffer(gpu, 3 * 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'rippleDem');
      this.cellBuf = CV.buffer(gpu, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'rippleCell');
      const n = t.near, f = t.far;
      gpu.queue.writeBuffer(this.gdBuf, 0, new Float32Array([n.x0, n.z0, n.nx * n.dx, n.nz * n.dz, f.x0, f.z0, f.nx * f.dx, f.nz * f.dz, t.blend, 0, 0, 0]));
      gpu.queue.writeBuffer(this.cellBuf, 0, new Float32Array([this.cell, 0, 0, 0]));
      this.samLin = dev.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
      const views = this.state.map(s => s.createView());
      this.bgStep = [0, 1].map(s => [0, 1].map(cur => dev.createBindGroup({ layout: this.pStep.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uBuf[s] } }, { binding: 1, resource: views[cur] }, { binding: 2, resource: views[1 - cur] },
        { binding: 3, resource: t.demNear.createView() }, { binding: 4, resource: t.demFar.createView() }, { binding: 5, resource: this.samLin },
        { binding: 6, resource: { buffer: this.gdBuf } }] })));
      this.bgOut = [0, 1].map(i => dev.createBindGroup({ layout: this.pOut.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.cellBuf } }, { binding: 1, resource: views[i] }, { binding: 2, resource: this.out.createView() }] }));
    }

    // World-space impulse (metres). strength = metres of water height (negative = dip).
    splash(x, z, radius = 0.17, strength = -0.05) {
      const c = this.center, h = this.size * 0.34;
      if (Math.abs(x - c[0]) > h || Math.abs(z - c[1]) > h) { this.center = [x, z]; this.forceClear = true; }
      if (this.pending.length < MAX_IMP) this.pending.push([x, z, radius, strength]);
    }
    // Records sim steps. 'focus' is a point the window should follow (a spot ahead of the camera).
    encode(encoder, dt, focus, seaLevel) {
      if (!this.enabled) return;
      const cs = this.cell;
      if (!this.forceClear && focus) {
        const c = this.center;
        if (Math.hypot(focus[0] - c[0], focus[1] - c[1]) > this.size * 0.14) this.center = [focus[0], focus[1]];
      }
      const org = [Math.round(this.center[0] / cs) - NR / 2, Math.round(this.center[1] / cs) - NR / 2];
      let old = this.oldOrigin || org;
      if (this.forceClear) { old = [org[0] + 100003, org[1] + 77]; this.forceClear = false; } // every texel reads as "new cell" -> cleared
      this.oldOrigin = org;
      const steps = 2, sdt = Math.min(dt, 1 / 30) / steps;
      const imps = this.pending.splice(0, MAX_IMP);
      for (let s = 0; s < steps; s++) {
        const u = new Float32Array((3 + MAX_IMP) * 4);
        u.set(s === 0 ? [org[0], org[1], old[0], old[1]] : [org[0], org[1], org[0], org[1]], 0);
        u.set([cs, sdt, this.params.damping, seaLevel], 4);
        u.set([s === 0 ? imps.length : 0, this.params.cMax, this.params.agitDecay, 0], 8);
        if (s === 0) for (let i = 0; i < imps.length; i++) u.set(imps[i], 12 + i * 4);
        this.gpu.queue.writeBuffer(this.uBuf[s], 0, u);
        const pass = encoder.beginComputePass({ label: 'ripple', timestampWrites: CV.tw('ripple') });
        pass.setPipeline(this.pStep); pass.setBindGroup(0, this.bgStep[s][this.cur]); pass.dispatchWorkgroups(NR / 8, NR / 8, 1);
        pass.end();
        this.cur ^= 1;
      }
      const p2 = encoder.beginComputePass({ label: 'rippleOut', timestampWrites: CV.tw('rippleOut') });
      p2.setPipeline(this.pOut); p2.setBindGroup(0, this.bgOut[this.cur]); p2.dispatchWorkgroups(NR / 8, NR / 8, 1);
      p2.end();
    }

    // Diagnostics: read back the render-side texture (h, dh/dx, dh/dz, agitation) and report extrema.
    async readStats() {
      const gpu = this.gpu, dev = gpu.device, f32 = gpu.fmtSim === 'rgba32float', bpp = f32 ? 16 : 8, bpr = NR * bpp;
      const buf = dev.createBuffer({ size: bpr * NR, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = dev.createCommandEncoder(); enc.copyTextureToBuffer({ texture: this.out }, { buffer: buf, bytesPerRow: bpr }, [NR, NR]); dev.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const raw = buf.getMappedRange(); let maxH = 0, maxAg = 0, n = 0, sumAbs = 0;
      const f16 = (h) => { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : e === 31 ? NaN : s * (1 + m / 1024) * 2 ** (e - 15); };
      const d = f32 ? new Float32Array(raw) : new Uint16Array(raw);
      for (let i = 0; i < NR * NR; i++) { const h = f32 ? d[i * 4] : f16(d[i * 4]), ag = f32 ? d[i * 4 + 3] : f16(d[i * 4 + 3]); if (Math.abs(h) > maxH) maxH = Math.abs(h); if (ag > maxAg) maxAg = ag; if (Math.abs(h) > 1e-5) { n++; sumAbs += Math.abs(h); } }
      buf.unmap(); buf.destroy();
      return { maxH, maxAg, activeCells: n, meanAbs: n ? sumAbs / n : 0, center: this.center, enabled: this.enabled };
    }
    uniforms() { return [this.center[0], this.center[1], this.size, this.enabled ? this.params.gain : 0]; }
  };
})();
