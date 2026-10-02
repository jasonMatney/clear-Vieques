// ClearVieques — temporal anti-aliasing. The projection is offset by a sub-pixel Halton jitter every frame; this pass reprojects last frame's
// result to the current pixel (world position from the depth buffer and the camera basis: with the infinite reverse-Z projection the view depth
// is simply near / d), clamps it to the spread of the current pixel's 3x3 neighbourhood (so moving water, glitter and disocclusions cannot
// smear), and blends in ~10% of the new frame. Blending happens on tone-compressed colour, so a bright glint does not dominate its neighbours.
// It runs at the internal render resolution, before grading and the edge-adaptive upscale. Per-pixel noise in the scene shaders (shadow-march
// jitter, the leafy skyline fringe) rotates with the frame index, so it averages out instead of showing as grain.
(function () {
  'use strict';
  const CV = window.CV;

  const code = (ms) => /* wgsl */`
struct TU {
  c0 : vec4<f32>, c1 : vec4<f32>, c2 : vec4<f32>, c3 : vec4<f32>,   // current camera: pos, near | fwd, tanX | right, tanY | up, -
  p0 : vec4<f32>, p1 : vec4<f32>, p2 : vec4<f32>, p3 : vec4<f32>,   // previous camera, same layout
  j : vec4<f32>,                                                    // jitter (NDC) x, y, blend weight of the new frame, reset
};
@group(0) @binding(0) var cur : texture_2d<f32>;
@group(0) @binding(1) var hist : texture_2d<f32>;
@group(0) @binding(2) var dep : ${ms ? 'texture_depth_multisampled_2d' : 'texture_depth_2d'};
@group(0) @binding(3) var samL : sampler;
@group(0) @binding(4) var<uniform> T : TU;
struct VOut { @builtin(position) pos : vec4<f32> };
@vertex fn vs(@builtin(vertex_index) vid : u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut; o.pos = vec4<f32>(p[vid], 0.0, 1.0); return o;
}
fn lum(c : vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn squash(c : vec3<f32>) -> vec3<f32> { return c / (1.0 + lum(c)); }                  // reversible tone compression for blending
fn unsquash(c : vec3<f32>) -> vec3<f32> { return c / max(1.0 - lum(c), 1e-3); }
fn toYCoCg(c : vec3<f32>) -> vec3<f32> { return vec3<f32>(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
fn fromYCoCg(c : vec3<f32>) -> vec3<f32> { return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }
// 5-tap Catmull-Rom history fetch: keeps the history sharp (bilinear would blur a little more every frame)
fn historyCR(uv : vec2<f32>, dims : vec2<f32>) -> vec3<f32> {
  let sp = uv * dims; let tc = floor(sp - 0.5) + 0.5; let f = sp - tc;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = (tc - 1.0) / dims; let t3 = (tc + 2.0) / dims; let t12 = (tc + w2 / w12) / dims;
  let r = textureSampleLevel(hist, samL, vec2<f32>(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y)
        + textureSampleLevel(hist, samL, vec2<f32>(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y)
        + textureSampleLevel(hist, samL, t12, 0.0).rgb * (w12.x * w12.y)
        + textureSampleLevel(hist, samL, vec2<f32>(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y)
        + textureSampleLevel(hist, samL, vec2<f32>(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
  let ws = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return max(r / ws, vec3<f32>(0.0));
}
@fragment fn fs(in : VOut) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(cur));
  let ip = vec2<i32>(in.pos.xy);
  let hi = vec2<i32>(dims) - vec2<i32>(1);
  let c = textureLoad(cur, ip, 0).rgb;
  if (T.j.w > 0.5) { return vec4<f32>(c, 1.0); }
  // neighbourhood mean and spread (YCoCg of the compressed colour): the history is clipped to mean +- 1.25 sigma
  var m1 = vec3<f32>(0.0); var m2 = vec3<f32>(0.0);
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    let s = toYCoCg(squash(textureLoad(cur, clamp(ip + vec2<i32>(x, y), vec2<i32>(0), hi), 0).rgb));
    m1 += s; m2 += s * s;
  } }
  let mean = m1 / 9.0; let sd = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
  // where was this surface last frame?
  let ndc = vec2<f32>(in.pos.x / dims.x * 2.0 - 1.0, 1.0 - in.pos.y / dims.y * 2.0) - T.j.xy;
  let d = textureLoad(dep, ip, 0);
  let ray = T.c1.xyz + T.c2.xyz * (ndc.x * T.c1.w) + T.c3.xyz * (ndc.y * T.c2.w);   // unit forward component
  var v = ray;                                                        // the sky (d = 0) reprojects by direction alone
  if (d > 1e-7) { v = T.c0.xyz + ray * (T.c0.w / d) - T.p0.xyz; }
  let z = dot(v, T.p1.xyz);
  let pc = vec2<f32>(dot(v, T.p2.xyz) / (z * T.p1.w), dot(v, T.p3.xyz) / (z * T.p2.w));
  let uvp = vec2<f32>(pc.x * 0.5 + 0.5, 0.5 - pc.y * 0.5);
  if (z <= 0.0 || uvp.x < 0.0 || uvp.y < 0.0 || uvp.x > 1.0 || uvp.y > 1.0) { return vec4<f32>(c, 1.0); }
  let h = clamp(toYCoCg(squash(historyCR(uvp, dims))), mean - 1.25 * sd, mean + 1.25 * sd);
  let o = mix(h, toYCoCg(squash(c)), T.j.z);
  return vec4<f32>(max(unsquash(fromYCoCg(o)), vec3<f32>(0.0)), 1.0);
}
`;

  const halton = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };

  CV.TAA = class TAA {
    constructor(gpu) {
      this.gpu = gpu; this.enabled = true; this.frame = 0; this.idx = 0; this.valid = false; this.prev = null;
      this.ubuf = CV.buffer(gpu, 9 * 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'taaU');
      this.u = new Float32Array(9 * 4);
      this.sampler = gpu.device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    }
    // pipeline for a depth buffer with this sample count (the scene's MSAA setting)
    pipeline(sampleCount) {
      if (this.pipe && this.ms === sampleCount) return;
      const dev = this.gpu.device, F = GPUShaderStage.FRAGMENT, ms = sampleCount > 1;
      this.ms = sampleCount;
      this.bgl = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: F, texture: { sampleType: 'float' } },
        { binding: 1, visibility: F, texture: { sampleType: 'float' } },
        { binding: 2, visibility: F, texture: { sampleType: 'depth', multisampled: ms } },
        { binding: 3, visibility: F, sampler: { type: 'filtering' } },
        { binding: 4, visibility: F, buffer: { type: 'uniform' } }] });
      const mod = dev.createShaderModule({ label: 'taa', code: code(ms) });
      this.pipe = dev.createRenderPipeline({ label: 'taa', layout: dev.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
        vertex: { module: mod, entryPoint: 'vs' }, fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] } });
      this.bg = null;
    }
    // history pair at the internal resolution; bind groups read (current HDR, the other history, depth)
    resize(w, h, hdrView, depthView) {
      const dev = this.gpu.device;
      for (const t of this.tex || []) t.destroy();
      this.tex = [0, 1].map(i => dev.createTexture({ label: 'taa' + i, size: [w, h], format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }));
      this.views = this.tex.map(t => t.createView());
      this.bg = [0, 1].map(i => dev.createBindGroup({ layout: this.bgl, entries: [
        { binding: 0, resource: hdrView }, { binding: 1, resource: this.views[1 - i] }, { binding: 2, resource: depthView },
        { binding: 3, resource: this.sampler }, { binding: 4, resource: { buffer: this.ubuf } }] }));
      this.valid = false;
    }
    // sub-pixel offset (NDC) for this frame: Halton (2, 3), 8-frame cycle
    jitter(w, h) {
      const k = (this.frame % 8) + 1;
      return [(halton(k, 2) - 0.5) * 1.5 / w, (halton(k, 3) - 0.5) * 1.5 / h];   // +-0.375 px: smooths edges without blurring the texture much
    }
    // cam = CV.Camera#matrices() of this frame (before jitter); returns the index of the history texture that now holds the result
    encode(enc, cam, jit, near) {
      const p = this.prev, u = this.u;
      // a jump (site switch, hero view, reset) starts the history over
      const reset = !this.valid || !p || Math.hypot(cam.pos[0] - p.pos[0], cam.pos[1] - p.pos[1], cam.pos[2] - p.pos[2]) > 25 ||
        (cam.fwd[0] * p.fwd[0] + cam.fwd[1] * p.fwd[1] + cam.fwd[2] * p.fwd[2]) < 0.9;
      const put = (o, c) => { u.set([...c.pos, near], o); u.set([...c.fwd, c.tanX], o + 4); u.set([...c.right, c.tanY], o + 8); u.set([...c.up, 0], o + 12); };
      put(0, cam); put(16, p || cam);
      u.set([jit[0], jit[1], 0.1, reset ? 1 : 0], 32);
      this.gpu.queue.writeBuffer(this.ubuf, 0, u);
      const i = this.idx;
      const pass = enc.beginRenderPass({ label: 'taa', timestampWrites: CV.tw('taa'), colorAttachments: [{ view: this.views[i], loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
      pass.setPipeline(this.pipe); pass.setBindGroup(0, this.bg[i]); pass.draw(3); pass.end();
      this.prev = { pos: cam.pos.slice(), fwd: cam.fwd.slice(), right: cam.right.slice(), up: cam.up.slice(), tanX: cam.tanX, tanY: cam.tanY };
      this.valid = true; this.idx = 1 - i; this.frame++;
      return i;
    }
  };
})();
