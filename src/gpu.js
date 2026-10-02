// ClearVieques — WebGPU bootstrap and small helpers.
(function () {
  'use strict';
  const CV = window.CV;

  CV.initGPU = async function (canvas) {
    if (!navigator.gpu) throw new Error('This browser does not support WebGPU. Use a recent Chrome or Edge, or Safari 26 or later (Mac, iPhone, iPad).');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('This browser supports WebGPU, but no usable graphics adapter was found (hardware acceleration may be turned off, or the GPU is blocklisted).');
    // ?compat=1 runs as a device without the optional features would (many phones, tablets and integrated GPUs): half-float DEM and
    // simulation textures, no GPU timing. Lets the fallback path be tested on any machine.
    const compat = new URLSearchParams(location.search).get('compat') === '1';
    const features = [];
    if (!compat) for (const f of ['float32-filterable', 'timestamp-query']) if (adapter.features.has(f)) features.push(f);
    const L = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(L.maxStorageBufferBindingSize, 1 << 28),
        maxBufferSize: Math.min(L.maxBufferSize, 1 << 28),
        maxTextureDimension2D: Math.min(L.maxTextureDimension2D, 8192),
      },
    });
    const gpu = {
      adapter, device, queue: device.queue, canvas,
      f32filter: features.includes('float32-filterable'),
      timestamps: features.includes('timestamp-query'),
      format: navigator.gpu.getPreferredCanvasFormat(),
      info: adapter.info || {},
    };
    device.addEventListener('uncapturederror', e => CV.log.error('WebGPU:', e.error && e.error.message));
    device.lost.then(i => CV.log.error('WebGPU device lost:', i.message));
    gpu.ctx = canvas.getContext('webgpu');
    gpu.ctx.configure({ device, format: gpu.format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    // Formats that depend on float32-filterable (Apple / NVIDIA / AMD all have it; Intel usually).
    gpu.fmtDem = gpu.f32filter ? 'r32float' : 'r16float';
    gpu.fmtSim = gpu.f32filter ? 'rgba32float' : 'rgba16float';
    return gpu;
  };

  // Compile WGSL and surface errors with source context (compile errors are the most common failure mode).
  CV.shader = async function (gpu, label, code) {
    const module = gpu.device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    let bad = false; const lines = code.split('\n');
    for (const m of info.messages) {
      const ctx = lines.slice(Math.max(0, m.lineNum - 2), m.lineNum + 1).map((l, i) => `${Math.max(1, m.lineNum - 1) + i}| ${l}`).join('\n');
      const text = `[${label}] ${m.type} @${m.lineNum}:${m.linePos}: ${m.message}\n${ctx}`;
      if (m.type === 'error') { bad = true; CV.log.error(text); } else CV.log.warn(text);
    }
    if (bad) throw new Error(`WGSL compile failed: ${label}`);
    return module;
  };

  // Run fn inside validation/out-of-memory error scopes and log anything raised.
  CV.scoped = async function (gpu, label, fn) {
    gpu.device.pushErrorScope('validation');
    gpu.device.pushErrorScope('out-of-memory');
    let r; try { r = await fn(); } finally {
      const oom = await gpu.device.popErrorScope(); const val = await gpu.device.popErrorScope();
      if (oom) CV.log.error(`[${label}] OOM: ${oom.message}`);
      if (val) CV.log.error(`[${label}] validation: ${val.message}`);
    }
    return r;
  };

  CV.buffer = function (gpu, size, usage, label) {
    return gpu.device.createBuffer({ label, size: Math.ceil(size / 4) * 4, usage });
  };



  // Per-pass GPU profiler. CV.prof is null in normal operation; while set, every pass created through CV.tw(label) records a timestamp pair.
  CV.prof = null; CV.tw = (label) => (CV.prof ? CV.prof.next(label) : undefined);
  CV.Profiler = class Profiler {
    constructor(gpu, maxPasses = 24) {
      this.gpu = gpu; this.max = maxPasses; this.labels = []; this.n = 0;
      this.qs = gpu.device.createQuerySet({ type: 'timestamp', count: maxPasses * 2 });
      this.resolve = gpu.device.createBuffer({ size: maxPasses * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.read = gpu.device.createBuffer({ size: maxPasses * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    next(label) { if (this.n >= this.max) return undefined; const i = this.n++; this.labels.push(label);
      return { querySet: this.qs, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 }; }
    finish(enc) { enc.resolveQuerySet(this.qs, 0, this.n * 2, this.resolve, 0); enc.copyBufferToBuffer(this.resolve, 0, this.read, 0, this.n * 16); }
    async results() {
      await this.read.mapAsync(GPUMapMode.READ, 0, this.n * 16); const t = new BigUint64Array(this.read.getMappedRange(0, this.n * 16).slice(0)); this.read.unmap();
      const out = {}; for (let i = 0; i < this.n; i++) out[this.labels[i]] = (out[this.labels[i]] || 0) + Number(t[i * 2 + 1] - t[i * 2]) / 1e6;
      return out;
    }
  };

  // Optional GPU frame timer using timestamp queries; results arrive a few frames late through a small ring of staging buffers.
  CV.GpuTimer = class GpuTimer {
    constructor(gpu) {
      this.gpu = gpu; this.enabled = gpu.timestamps; this.avg = 0; this.samples = 0; this.slot = 0; this.pending = [false, false, false];
      if (!this.enabled) return;
      const dev = gpu.device;
      this.qs = dev.createQuerySet({ type: 'timestamp', count: 2 });
      this.resolve = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.stage = [0, 1, 2].map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    }
    first() { return this.enabled ? { querySet: this.qs, beginningOfPassWriteIndex: 0 } : undefined; }
    last() { return this.enabled ? { querySet: this.qs, endOfPassWriteIndex: 1 } : undefined; }
    // call before submit (after the last pass has ended)
    resolveInto(encoder) {
      if (!this.enabled || this.pending[this.slot]) { this.skip = true; return; }
      this.skip = false;
      encoder.resolveQuerySet(this.qs, 0, 2, this.resolve, 0);
      encoder.copyBufferToBuffer(this.resolve, 0, this.stage[this.slot], 0, 16);
    }
    // call after submit
    readback() {
      if (!this.enabled || this.skip) return;
      const s = this.slot; this.slot = (this.slot + 1) % 3; this.pending[s] = true;
      this.stage[s].mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigUint64Array(this.stage[s].getMappedRange().slice(0));
        this.stage[s].unmap(); this.pending[s] = false;
        const ms = Number(t[1] - t[0]) / 1e6;
        if (ms > 0 && ms < 200) { this.avg = this.samples === 0 ? ms : this.avg * 0.92 + ms * 0.08; this.samples++; }
      }).catch(() => { this.pending[s] = false; });
    }
  };

  // Generic 2x2 box mip generator for rgba16float 2D / 2D-array textures (compute, write-only storage).
  CV.MipGen = class MipGen {
    constructor(gpu) { this.gpu = gpu; this.ready = this.init(); }
    async init() {
      const gpu = this.gpu;
      const code = `
@group(0) @binding(0) var src : texture_2d_array<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let sz = textureDimensions(dst);
  if (id.x >= sz.x || id.y >= sz.y) { return; }
  let b = vec2<i32>(id.xy) * 2;
  let l = i32(id.z);
  let c = (textureLoad(src, b, l, 0) + textureLoad(src, b + vec2<i32>(1, 0), l, 0) +
           textureLoad(src, b + vec2<i32>(0, 1), l, 0) + textureLoad(src, b + vec2<i32>(1, 1), l, 0)) * 0.25;
  textureStore(dst, vec2<i32>(id.xy), l, c);
}`;
      const module = await CV.shader(gpu, 'mipgen', code);
      this.pipeline = gpu.device.createComputePipeline({ label: 'mipgen', layout: 'auto', compute: { module, entryPoint: 'main' } });
    }
    // Build per-level bind groups for a texture created with mipLevelCount>1.
    makePasses(tex, width, height, layers) {
      const levels = CV.mipCount(width, height), passes = [];
      for (let l = 1; l < levels; l++) {
        const w = Math.max(1, width >> l), h = Math.max(1, height >> l);
        const srcV = tex.createView({ dimension: '2d-array', baseMipLevel: l - 1, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: layers });
        const dstV = tex.createView({ dimension: '2d-array', baseMipLevel: l, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: layers });
        const bg = this.gpu.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: srcV }, { binding: 1, resource: dstV }] });
        passes.push({ bg, gx: Math.ceil(w / 8), gy: Math.ceil(h / 8), gz: layers });
      }
      return passes;
    }
    encode(pass, passes) {
      pass.setPipeline(this.pipeline);
      for (const p of passes) { pass.setBindGroup(0, p.bg); pass.dispatchWorkgroups(p.gx, p.gy, p.gz); }
    }
  };
})();
