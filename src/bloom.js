// ClearVieques — bloom: the soft glow a real lens and eye put around very bright light (sun glints, the sun's disc, the moon's path on the water).
// Only light above a threshold takes part (set where the tone mapper starts to saturate, so it follows the exposure): the first downsample keeps
// the excess over it (soft knee), a chain of downsamples and tent-filtered upsamples spreads it, and grading adds it back. Ordinary surfaces are
// untouched. The first downsample also caps each tap so one extreme pixel cannot flicker as a blob.
(function () {
  'use strict';
  const CV = window.CV;
  const LEVELS = 6;

  const code = /* wgsl */`
@group(0) @binding(0) var tA : texture_2d<f32>;
@group(0) @binding(1) var samB : sampler;
@group(0) @binding(2) var tB : texture_2d<f32>;
@group(0) @binding(3) var<uniform> BU : vec4<f32>;              // threshold (scene radiance), soft knee, cap, -
struct VOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vid : u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut; o.pos = vec4<f32>(p[vid], 0.0, 1.0); o.uv = vec2<f32>(p[vid].x * 0.5 + 0.5, 0.5 - p[vid].y * 0.5); return o;
}
fn tap(uv : vec2<f32>, first : bool) -> vec3<f32> {
  let c = textureSampleLevel(tA, samB, uv, 0.0).rgb;
  if (!first) { return c; }
  let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  let k = BU.y;
  var soft = clamp(l - BU.x + k, 0.0, 2.0 * k);
  soft = soft * soft / (4.0 * k + 1e-5);
  let over = max(soft, l - BU.x);                                  // the part of the luminance above the threshold
  return c * (over / max(l, 1e-5)) * min(1.0, BU.z / max(l, 1e-4));
}
// 13-tap downsample (a 4x4 box made of overlapping bilinear fetches): no aliasing or shimmer as highlights move
fn down13(uv : vec2<f32>, first : bool) -> vec3<f32> {
  let t = 1.0 / vec2<f32>(textureDimensions(tA));
  let e = tap(uv, first);
  let a = tap(uv + t * vec2<f32>(-2.0, -2.0), first); let b = tap(uv + t * vec2<f32>(0.0, -2.0), first); let c = tap(uv + t * vec2<f32>(2.0, -2.0), first);
  let d = tap(uv + t * vec2<f32>(-2.0, 0.0), first);                                                    let f = tap(uv + t * vec2<f32>(2.0, 0.0), first);
  let g = tap(uv + t * vec2<f32>(-2.0, 2.0), first); let h = tap(uv + t * vec2<f32>(0.0, 2.0), first); let i = tap(uv + t * vec2<f32>(2.0, 2.0), first);
  let j = tap(uv + t * vec2<f32>(-1.0, -1.0), first); let k = tap(uv + t * vec2<f32>(1.0, -1.0), first);
  let l = tap(uv + t * vec2<f32>(-1.0, 1.0), first); let m = tap(uv + t * vec2<f32>(1.0, 1.0), first);
  return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}
@fragment fn fs_down0(in : VOut) -> @location(0) vec4<f32> { return vec4<f32>(down13(in.uv, true), 1.0); }
@fragment fn fs_down(in : VOut) -> @location(0) vec4<f32> { return vec4<f32>(down13(in.uv, false), 1.0); }
// this level (tA) plus the tent-filtered coarser level (tB) above it
@fragment fn fs_up(in : VOut) -> @location(0) vec4<f32> {
  let t = 1.0 / vec2<f32>(textureDimensions(tB));
  var s = textureSampleLevel(tB, samB, in.uv, 0.0).rgb * 4.0;
  s += (textureSampleLevel(tB, samB, in.uv + vec2<f32>(-t.x, 0.0), 0.0).rgb + textureSampleLevel(tB, samB, in.uv + vec2<f32>(t.x, 0.0), 0.0).rgb
      + textureSampleLevel(tB, samB, in.uv + vec2<f32>(0.0, -t.y), 0.0).rgb + textureSampleLevel(tB, samB, in.uv + vec2<f32>(0.0, t.y), 0.0).rgb) * 2.0;
  s += textureSampleLevel(tB, samB, in.uv - t, 0.0).rgb + textureSampleLevel(tB, samB, in.uv + t, 0.0).rgb
     + textureSampleLevel(tB, samB, in.uv + vec2<f32>(t.x, -t.y), 0.0).rgb + textureSampleLevel(tB, samB, in.uv + vec2<f32>(-t.x, t.y), 0.0).rgb;
  return vec4<f32>(textureSampleLevel(tA, samB, in.uv, 0.0).rgb + s / 16.0, 1.0);
}
`;

  CV.Bloom = class Bloom {
    constructor(gpu) {
      const dev = gpu.device, F = GPUShaderStage.FRAGMENT;
      this.gpu = gpu;
      this.sampler = dev.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
      this.bgl = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: F, texture: { sampleType: 'float' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, texture: { sampleType: 'float' } },
        { binding: 3, visibility: F, buffer: { type: 'uniform' } }] });
      this.ubuf = CV.buffer(gpu, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'bloomU');
      const mod = dev.createShaderModule({ label: 'bloom', code });
      const lay = dev.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
      const pipe = (fs) => dev.createRenderPipeline({ label: 'bloom-' + fs, layout: lay, vertex: { module: mod, entryPoint: 'vs' },
        fragment: { module: mod, entryPoint: fs, targets: [{ format: 'rgba16float' }] } });
      this.pDown0 = pipe('fs_down0'); this.pDown = pipe('fs_down'); this.pUp = pipe('fs_up');
    }
    // sources: the HDR views bloom may read. Builds the down (D) and up (U) chains at half resolution and below; this.view (U level 0) is the
    // sum of the spread excess light over all levels, which grading adds back (divided by this.n).
    resize(w, h, sources) {
      const dev = this.gpu.device, usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
      for (const t of [this.D, this.U]) if (t) t.destroy();
      const w0 = Math.max(1, w >> 1), h0 = Math.max(1, h >> 1);
      const n = Math.max(2, Math.min(LEVELS, Math.floor(Math.log2(Math.min(w0, h0))) - 1));
      this.n = n;
      this.D = dev.createTexture({ label: 'bloomDown', size: [w0, h0], format: 'rgba16float', mipLevelCount: n, usage });
      this.U = dev.createTexture({ label: 'bloomUp', size: [w0, h0], format: 'rgba16float', mipLevelCount: n - 1, usage });
      const mv = (t, i) => t.createView({ baseMipLevel: i, mipLevelCount: 1 });
      this.dv = [...Array(n).keys()].map(i => mv(this.D, i));
      this.uv = [...Array(n - 1).keys()].map(i => mv(this.U, i));
      const bg = (a, b) => dev.createBindGroup({ layout: this.bgl, entries: [{ binding: 0, resource: a }, { binding: 1, resource: this.sampler },
        { binding: 2, resource: b }, { binding: 3, resource: { buffer: this.ubuf } }] });
      this.bgSrc = sources.map(s => bg(s, this.dv[n - 1]));                 // binding 2 is unused by the down shaders: any other subresource
      this.bgDown = [...Array(n).keys()].map(i => i === 0 ? null : bg(this.dv[i - 1], this.uv[0]));
      // U[i] = D[i] + tent(coarser): the coarsest up level reads D[n-1], the others read U[i+1]
      this.bgUp = [...Array(n - 1).keys()].map(i => bg(this.dv[i], i === n - 2 ? this.dv[n - 1] : this.uv[i + 1]));
      this.view = this.uv[0];
    }
    // threshold: scene radiance above which light glows (the caller derives it from the exposure); cap: the most one pixel may contribute
    encode(enc, srcIndex, threshold, cap) {
      this.gpu.queue.writeBuffer(this.ubuf, 0, new Float32Array([threshold, threshold * 0.5, cap, 0]));
      const pass = (view, pipe, group, label) => {
        const p = enc.beginRenderPass({ label, colorAttachments: [{ view, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
        p.setPipeline(pipe); p.setBindGroup(0, group); p.draw(3); p.end();
      };
      pass(this.dv[0], this.pDown0, this.bgSrc[srcIndex], 'bloomDown0');
      for (let i = 1; i < this.n; i++) pass(this.dv[i], this.pDown, this.bgDown[i], 'bloomDown');
      for (let i = this.n - 2; i >= 0; i--) pass(this.uv[i], this.pUp, this.bgUp[i], 'bloomUp');
    }
  };
})();
