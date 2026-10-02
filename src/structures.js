// ClearVieques — OpenStreetMap picnic shelters drawn as small instanced meshes (slab, four posts, hipped metal roof, a picnic table).
// Footprints, centroids and orientations come from the sparse OSM extract in data/ (Playa Caracas has three; Mosquito Bay none). Each shelter
// also casts a soft analytic roof shadow onto the ground (see structShadow in the shared prelude), so the sand beneath is not sunlit.
(function () {
  'use strict';
  const CV = window.CV;
  const MAX = 8, SIZE = 4.3;   // shelters per site; nominal footprint edge (m) the mesh is built for

  // ---- geometry: boxes and a hip roof as triangles (pos.xyz, normal.xyz, material id, pad); local frame: x, z on the ground, y up
  function build() {
    const v = [];
    const tri = (a, b, c, n, m) => { for (const p of [a, b, c]) v.push(p[0], p[1], p[2], n[0], n[1], n[2], m, 0); };
    const quad = (a, b, c, d, n, m) => { tri(a, b, c, n, m); tri(a, c, d, n, m); };
    const box = (cx, cy, cz, sx, sy, sz, m) => {
      const x0 = cx - sx / 2, x1 = cx + sx / 2, y0 = cy - sy / 2, y1 = cy + sy / 2, z0 = cz - sz / 2, z1 = cz + sz / 2;
      quad([x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [0, 1, 0], m);
      quad([x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [0, -1, 0], m);
      quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], [1, 0, 0], m);
      quad([x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0], [-1, 0, 0], m);
      quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], m);
      quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], m);
    };
    box(0, 0.07, 0, 4.7, 0.14, 4.7, 2);                                  // slab
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) box(sx * 2.0, 1.39, sz * 2.0, 0.17, 2.5, 0.17, 0);   // posts
    // hip roof: eaves 5.6 m square at y = 2.65, ridge 1.2 m square at y = 3.95
    const E = 2.8, T = 0.6, y0 = 2.65, y1 = 3.95;
    const e = [[-E, y0, -E], [E, y0, -E], [E, y0, E], [-E, y0, E]], t = [[-T, y1, -T], [T, y1, -T], [T, y1, T], [-T, y1, T]];
    const slope = (a, b, c, d) => {   // outward normal from the edge vectors
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
      let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]; const l = Math.hypot(...n); n = n.map(x => x / l);
      if (n[1] < 0) n = n.map(x => -x);
      quad(a, b, c, d, n, 1);
    };
    slope(e[0], e[1], t[1], t[0]); slope(e[1], e[2], t[2], t[1]); slope(e[2], e[3], t[3], t[2]); slope(e[3], e[0], t[0], t[3]);
    quad(e[0], e[1], e[2], e[3], [0, -1, 0], 3);                         // roof underside (dark timber)
    // picnic table
    box(0, 0.76, 0, 2.0, 0.06, 0.8, 0);
    for (const s of [-1, 1]) { box(0, 0.46, s * 0.7, 2.0, 0.05, 0.28, 0); box(0.8, 0.40, s * 0.7, 0.06, 0.4, 0.06, 0); box(-0.8, 0.40, s * 0.7, 0.06, 0.4, 0.06, 0); }
    return new Float32Array(v);
  }

  const wgsl = () => CV.wgsl.prelude() + /* wgsl */`
struct SU { inst : array<vec4<f32>, ${MAX * 2}> };     // per shelter: (x, ground y, z, yaw), (scale, -, -, -)
@group(1) @binding(0) var<uniform> SI : SU;
struct SV { @location(0) pos : vec3<f32>, @location(1) nrm : vec3<f32>, @location(2) mat : f32 };
struct SOut { @builtin(position) p : vec4<f32>, @location(0) w : vec3<f32>, @location(1) n : vec3<f32>, @location(2) mat : f32 };

@vertex fn vs_struct(v : SV, @builtin(instance_index) ii : u32) -> SOut {
  let a = SI.inst[ii * 2u]; let sc = SI.inst[ii * 2u + 1u].x;
  let c = cos(a.w); let s = sin(a.w);
  let pl = v.pos * sc;
  let wp = vec3<f32>(a.x + pl.x * c - pl.z * s, a.y + pl.y, a.z + pl.x * s + pl.z * c);
  var o : SOut;
  o.w = wp; o.mat = v.mat;
  o.n = vec3<f32>(v.nrm.x * c - v.nrm.z * s, v.nrm.y, v.nrm.x * s + v.nrm.z * c);
  o.p = G.viewProj * vec4<f32>(wp, 1.0);
  return o;
}

@fragment fn fs_struct(in : SOut) -> @location(0) vec4<f32> {
  let toCam = G.camPos.xyz - in.w; let dist = length(toCam); let V = toCam / dist;
  var n = normalize(in.n); if (dot(n, V) < 0.0 && in.mat > 2.5) { n = -n; }
  var alb = vec3<f32>(0.20, 0.14, 0.09);                                       // weathered timber
  if (in.mat > 0.5 && in.mat < 1.5) { alb = vec3<f32>(0.22, 0.078, 0.052); }     // weathered red corrugated roof
  if (in.mat > 1.5 && in.mat < 2.5) { alb = vec3<f32>(0.42, 0.41, 0.39); }      // concrete slab
  if (in.mat > 2.5) { alb = vec3<f32>(0.11, 0.075, 0.05); }                     // roof underside
  let L = G.sunDir.xyz;
  let ndl = max(dot(n, L), 0.0);
  let sh = sunShadow(in.w + n * 0.05, hash21(in.p.xy) * 0.5);
  let up = 0.5 + 0.5 * n.y;
  var col = alb / PI * (G.sunE.rgb * ndl * sh + G.skyE.rgb * up * mix(0.85, 0.45, step(2.5, in.mat)));
  col = applyFog(col, dist, -V);
  return vec4<f32>(col, 1.0);
}
`;

  CV.Structures = class Structures {
    constructor(gpu) {
      this.gpu = gpu; this.count = 0; this.slots = new Float32Array(MAX * 4);   // (x, z, yaw, roof half extent) per shelter, for the ground shadow
      this.verts = build(); this.vertexCount = this.verts.length / 8;
      const dev = gpu.device;
      this.vbuf = dev.createBuffer({ label: 'structVerts', size: this.verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      gpu.queue.writeBuffer(this.vbuf, 0, this.verts);
      this.ubuf = CV.buffer(gpu, MAX * 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'structInst');
      this.layout = dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }] });
      this.bg = dev.createBindGroup({ layout: this.layout, entries: [{ binding: 0, resource: { buffer: this.ubuf } }] });
      this.ready = CV.shader(gpu, 'structures', wgsl()).then(m => { this.module = m; });
    }

    // Place the shelters listed in a terrain's OSM extract on the real ground.
    setSite(terrain) {
      const list = ((terrain.osm && terrain.osm.structures) || []).slice(0, MAX), inst = new Float32Array(MAX * 8);
      this.slots.fill(0); this.count = list.length;
      list.forEach((s, i) => {
        const p = s.pts || [], c = s.centroid || [0, 0];
        let yaw = 0, side = SIZE;
        if (p.length >= 3) { yaw = Math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]); side = Math.sqrt(Math.max(s.area_m2 || SIZE * SIZE, 4)); }
        const sc = CV.clamp(side / SIZE, 0.8, 1.3);
        inst.set([c[0], terrain.heightAt(c[0], c[1]), c[1], yaw, sc, 0, 0, 0], i * 8);
        this.slots.set([c[0], c[1], yaw, 2.8 * sc], i * 4);
      });
      this.gpu.queue.writeBuffer(this.ubuf, 0, inst);
    }

    pipeline(sceneLayout, sampleCount) {
      const dev = this.gpu.device;
      return dev.createRenderPipeline({ label: 'structures', layout: dev.createPipelineLayout({ bindGroupLayouts: [sceneLayout, this.layout] }),
        vertex: { module: this.module, entryPoint: 'vs_struct', buffers: [{ arrayStride: 32, attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }, { shaderLocation: 2, offset: 24, format: 'float32' }] }] },
        fragment: { module: this.module, entryPoint: 'fs_struct', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, multisample: { count: sampleCount },
        depthStencil: { format: 'depth32float', depthCompare: 'greater', depthWriteEnabled: true } });
    }
  };
})();
