// ClearVieques — tree canopy bake. CUDEM is bare-earth, so the forest is procedural: a compute pass classifies the real terrain (slope, shore
// distance, curvature, elevation, sand / rock / grass / scrub) with the same WGSL the terrain shader uses, drops jittered crown centres on a
// Worley lattice where trees can grow, and writes
//   crownTex : 2 m texels — nearest crown centre (offset x, z), crown height, random id   (rgba8unorm)
//   meanTex  : 4 m texels on the near-DEM grid — stand mean canopy height, tree fraction  (rgba8unorm)
// The terrain vertex shader lifts the near mesh by the crown heights (tree-shaped skylines), the fragment shader shades each crown as a lit dome,
// and sun shadows / reflected rays see the smoothed stand height. Runs once per site (a few ms on the GPU).
(function () {
  'use strict';
  const CV = window.CV;

  const bakeCode = () => CV.wgsl.prelude() + CV.wgsl.materials + /* wgsl */`
struct BU { a : vec4<f32>, b : vec4<f32> };     // a: grid x0, z0, cell (m), nx ; b: nz, crown lattice (m), max canopy height (m), -
@group(1) @binding(0) var<uniform> B : BU;
@group(1) @binding(1) var outTex : texture_storage_2d<rgba8unorm, write>;

// (stand height in metres, tree weight 0..1) at a point of the bare-earth terrain
fn standAt(p : vec2<f32>) -> vec2<f32> {
  let ground = heightAt(p);
  if (ground < seaLevel() - 0.25) { return vec2<f32>(0.0); }
  let n = terrainNormal(p, 4.0);
  let aux = auxAt(p);
  let c = landClass(p, n, vec4<f32>(aux.x - 9.0, aux.y, aux.z, aux.w), 0.0);   // trees keep 9 m clear of the sand line the fragment shader will draw
  if (ground < seaLevel() + 0.35 && c.mangW < 0.5) { return vec2<f32>(0.0); }  // only mangroves stand in the intertidal shallows
  var tree = c.vegW * (1.0 - 0.92 * c.plain);
  tree *= 1.0 - 0.7 * c.rockW;
  let coastal = 1.0 - smoothstep(20.0, 120.0, aux.x);
  var h = mix(5.6, 3.0, coastal);                          // dry forest ~7 m inland, wind-pruned scrub near the shore
  h *= 0.78 + 0.44 * c.n2.g;                               // stand-scale variation
  h *= 1.0 + 0.35 * clamp(aux.y / 3.0, -0.6, 1.0);         // taller in gullies (concave), shorter on convex ridges
  h *= 1.0 - 0.30 * smoothstep(60.0, 130.0, c.elev);       // exposed high ground
  h = mix(h, 2.8, c.scrub * 0.8);                          // sea-grape band behind the sand
  h = mix(h, (3.6 + 2.6 * c.n2.g) * (1.0 + 0.15 * c.n3.r), c.mangW);   // mangrove: a lower, closed canopy
  tree = max(tree, c.mangW * 1.05);
  return vec2<f32>(h, tree);
}

@compute @workgroup_size(8, 8, 1)
fn bakeCrowns(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(B.a.w) || id.y >= u32(B.b.x)) { return; }
  let pos = B.a.xy + (vec2<f32>(id.xy) + 0.5) * B.a.z;
  let cell = B.b.y;
  let cf = crownField(pos, cell);                          // zw: vector to the nearest crown centre, in lattice cells
  let centre = pos + cf.zw * cell;
  let st = standAt(centre);
  let exists = st.y > 0.42 + 0.20 * (fract(cf.y * 3.3) - 0.5);   // ragged forest edges: whole crowns are in or out
  var H = 0.0;
  if (exists) {
    H = st.x * (0.72 + 0.56 * fract(cf.y * 13.7));
    // a crown standing at the sand line ends where the sand starts: no foliage height over texels that are themselves beach
    let own = landClass(pos, terrainNormal(pos, 4.0), auxAt(pos), 0.0);
    H *= 1.0 - smoothstep(0.02, 0.35, own.sandW);
  }
  let off = (centre - pos) / CROWN_OFF + 0.5;
  textureStore(outTex, vec2<i32>(id.xy), vec4<f32>(clamp(off, vec2<f32>(0.0), vec2<f32>(1.0)), clamp(H / B.b.z, 0.0, 1.0), cf.y));
}

@compute @workgroup_size(8, 8, 1)
fn bakeMean(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(B.a.w) || id.y >= u32(B.b.x)) { return; }
  let cell = B.a.z;
  let org = B.a.xy + vec2<f32>(id.xy) * cell;
  var sh = 0.0; var cnt = 0.0;
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      let q = org + (vec2<f32>(f32(i), f32(j)) + 0.5) * (cell * 0.25);
      let cr = crownAt(q);
      sh += crownH(q, cr, 1.0);
      cnt += select(0.0, 1.0, cr.H > 0.5);
    }
  }
  textureStore(outTex, vec2<i32>(id.xy), vec4<f32>(sh / 16.0 / B.b.z, cnt / 16.0, 0.0, 0.0));
}
`;

  CV.Canopy = class Canopy {
    constructor(gpu, sceneLayout) {
      this.gpu = gpu; this.sceneLayout = sceneLayout;
      this.lattice = 9.0;      // crown lattice (m): mean crown spacing
      this.ready = this.init();
    }
    async init() {
      const gpu = this.gpu, dev = gpu.device, C = GPUShaderStage.COMPUTE;
      this.l1 = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, storageTexture: { access: 'write-only', format: 'rgba8unorm' } }] });
      const mod = await CV.shader(gpu, 'canopyBake', bakeCode());
      const lay = dev.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.l1] });
      this.pCrowns = dev.createComputePipeline({ label: 'bakeCrowns', layout: lay, compute: { module: mod, entryPoint: 'bakeCrowns' } });
      this.pMean = dev.createComputePipeline({ label: 'bakeMean', layout: lay, compute: { module: mod, entryPoint: 'bakeMean' } });
      // 1x1 stand-ins for every scene binding the bake does not read (a texture may not be sampled and stored in one dispatch)
      const mk = (fmt, layers) => dev.createTexture({ size: [1, 1, layers || 1], format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING });
      this.dummy = { f16: mk('rgba16float').createView(), f16a: mk('rgba16float', 1).createView({ dimension: '2d-array' }), u8: mk('rgba8unorm').createView() };
    }

    // Bakes terrain.crownTex / terrain.meanTex. res = { samLin, samRep, noise } (renderer resources shared by the scene bind group).
    bake(terrain, res) {
      const gpu = this.gpu, dev = gpu.device, t = terrain, n = t.near, cr = t.crown, d = this.dummy;
      const G = new CV.UniformBlock(CV.GLOBALS);
      G.set('demNear', n.x0, n.z0, n.nx * n.dx, n.nz * n.dz);
      G.set('demFar', t.far.x0, t.far.z0, t.far.nx * t.far.dx, t.far.nz * t.far.dz);
      G.set('demInfo', t.blend, 0, 0, 0);
      G.set('kAbs', 0, 0, 0, t.seaLevel);
      G.set('crownA', cr.x0, cr.z0, cr.texel, cr.nx);
      G.set('crownB', cr.nz, n.dx, cr.maxH, 0);
      G.set('misc', 1, 0, 0, t.site && t.site.matSet ? 1 : 0);
      const gBuf = CV.buffer(gpu, G.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'bakeGlobals');
      gpu.queue.writeBuffer(gBuf, 0, G.data);
      const sceneBG = (crownView) => dev.createBindGroup({ layout: this.sceneLayout, entries: [
        { binding: 0, resource: { buffer: gBuf } }, { binding: 1, resource: res.samLin }, { binding: 2, resource: res.samRep },
        { binding: 3, resource: t.demNear.createView() }, { binding: 4, resource: t.demFar.createView() }, { binding: 5, resource: t.auxTex.createView() },
        { binding: 6, resource: d.f16 }, { binding: 7, resource: d.f16a }, { binding: 8, resource: d.f16a }, { binding: 9, resource: d.f16 }, { binding: 10, resource: d.f16 },
        { binding: 11, resource: res.noise.createView() }, { binding: 12, resource: crownView }, { binding: 13, resource: d.u8 }] });
      const ub = (a, b) => { const buf = CV.buffer(gpu, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'bakeU'); gpu.queue.writeBuffer(buf, 0, new Float32Array([...a, ...b])); return buf; };
      const lattice = (t.site && t.site.matSet === 1) ? 7.0 : this.lattice;   // closed mangrove canopy needs a denser crown lattice
      const u1 = ub([cr.x0, cr.z0, cr.texel, cr.nx], [cr.nz, lattice, cr.maxH, 0]);
      const u2 = ub([n.x0, n.z0, n.dx, n.nx], [n.nz, 0, cr.maxH, 0]);
      const g1 = (u, tex) => dev.createBindGroup({ layout: this.l1, entries: [{ binding: 0, resource: { buffer: u } }, { binding: 1, resource: tex.createView() }] });
      const enc = dev.createCommandEncoder({ label: 'canopyBake' });
      let p = enc.beginComputePass({ label: 'bakeCrowns' });
      p.setPipeline(this.pCrowns); p.setBindGroup(0, sceneBG(d.u8)); p.setBindGroup(1, g1(u1, t.crownTex));
      p.dispatchWorkgroups(Math.ceil(cr.nx / 8), Math.ceil(cr.nz / 8), 1); p.end();
      p = enc.beginComputePass({ label: 'bakeMean' });
      p.setPipeline(this.pMean); p.setBindGroup(0, sceneBG(t.crownTex.createView())); p.setBindGroup(1, g1(u2, t.meanTex));
      p.dispatchWorkgroups(Math.ceil(n.nx / 8), Math.ceil(n.nz / 8), 1); p.end();
      dev.queue.submit([enc.finish()]);
      t.canopyBaked = true;
    }
  };
})();
