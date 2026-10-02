// ClearVieques — tree canopy bake. CUDEM is bare-earth, so the forest is procedural: a compute pass classifies the real terrain (slope, shore
// distance, curvature, elevation, sand / rock / grass / scrub) with the same WGSL the terrain shader uses, drops jittered crown centres on a
// Worley lattice where trees can grow, and writes
//   cellTex  : one texel per lattice cell — crown height if a tree stands there, 0 if not (kept as terrain.cells for the 3D trees)
//   crownTex : 2 m texels — the crown that is tallest at this texel among the 3x3 neighbouring cells: offset to its centre (x, z), height, id
//   meanTex  : 4 m texels on the near-DEM grid — stand mean canopy height, tree fraction  (rgba8unorm)
// Taking the tallest neighbour (rather than the nearest centre) lets a crown's dome spill across its cell boundary where the neighbouring cell
// has no tree, so stands end in rounded edges instead of 2 m-stepped walls. The terrain vertex shader lifts the near mesh by the crown heights
// (tree-shaped skylines), the fragment shader shades each crown, and sun shadows / reflected rays see the smoothed stand height. Runs once per
// site (a few ms on the GPU).
(function () {
  'use strict';
  const CV = window.CV;

  const bakeCode = () => CV.wgsl.prelude() + CV.wgsl.materials + /* wgsl */`
struct BU { a : vec4<f32>, b : vec4<f32>, c : vec4<f32> };   // a: grid x0, z0, cell (m), nx ; b: nz, crown lattice (m), max canopy height (m), - ;
                                                             // c: first lattice cell i, j, cell count i, j (crowns) | swell, wind-sea travel angle (rad), -, - (mean)
@group(1) @binding(0) var<uniform> B : BU;
@group(1) @binding(1) var outTex : texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var cellTex : texture_2d<f32>;
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
  tree *= 1.0 - shoreRockFrom(p, aux.x, coastSteep(p));            // no trees on the rocky shore band of headlands and islets
  return vec2<f32>(h, tree);
}

// lattice cell g (integer coordinates, in lattice units) -> crown centre (m) and id: the same jitter and hash as crownField
fn cellCentre(g : vec2<f32>, lat : f32) -> vec2<f32> { return (g + 0.2 + 0.6 * hash22(g)) * lat; }
fn cellId(g : vec2<f32>) -> f32 { return hash21(g + vec2<f32>(3.7, 1.3)); }

@compute @workgroup_size(8, 8, 1)
fn bakeCells(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(B.c.z) || id.y >= u32(B.c.w)) { return; }
  let g = B.c.xy + vec2<f32>(id.xy);
  let cid = cellId(g);
  let st = standAt(cellCentre(g, B.b.y));
  let exists = st.y > 0.42 + 0.20 * (fract(cid * 3.3) - 0.5);   // ragged forest edges: whole crowns are in or out
  let H = select(0.0, st.x * (0.72 + 0.56 * fract(cid * 13.7)), exists);
  textureStore(outTex, vec2<i32>(id.xy), vec4<f32>(clamp(H / B.b.z, 0.0, 1.0), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn bakeCrowns(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(B.a.w) || id.y >= u32(B.b.x)) { return; }
  let pos = B.a.xy + (vec2<f32>(id.xy) + 0.5) * B.a.z;
  let lat = B.b.y;
  let i = floor(pos / lat);
  var best = 0.0; var bc = pos; var bH = 0.0; var bid = 0.0;
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    let g = i + vec2<f32>(f32(x), f32(y));
    let t = vec2<i32>(g - B.c.xy);
    if (t.x < 0 || t.y < 0 || t.x >= i32(B.c.z) || t.y >= i32(B.c.w)) { continue; }
    let H = textureLoad(cellTex, t, 0).r * B.b.z;
    if (H <= 0.0) { continue; }
    let c = cellCentre(g, lat);
    if (abs(c.x - pos.x) > 0.49 * CROWN_OFF || abs(c.y - pos.y) > 0.49 * CROWN_OFF) { continue; }   // must fit the stored offset
    let cid = cellId(g);
    let sc = H * crownProfile(crownRad(pos, crownShape(c, H, cid)), 1.0);
    if (sc > best) { best = sc; bc = c; bH = H; bid = cid; }
  } }
  let off = (bc - pos) / CROWN_OFF + 0.5;
  textureStore(outTex, vec2<i32>(id.xy), vec4<f32>(clamp(off, vec2<f32>(0.0), vec2<f32>(1.0)), clamp(bH / B.b.z, 0.0, 1.0), bid));
}

// Open water from p toward where the waves come from (m): march outward, ignore land until the ray has reached water (p itself may be on the
// beach), then stop at the first land. Islets, headlands and the far shore of the bay cast a "wave shadow".
fn openFetch(p : vec2<f32>, src : vec2<f32>) -> f32 {
  let sea = seaLevel();
  var t = 10.0; var wet = false;
  for (var k = 0; k < 24; k++) {
    let land = heightAt(p + src * t) > sea + 0.2;
    if (!land) { wet = true; } else if (wet || t > 80.0) { return t; }
    t *= 1.28;
  }
  return 6000.0;
}
// 0 (sheltered) .. 1 (open to the waves), over a ~70 degree fan around the direction the waves come from (waves spread into a shadow zone)
fn exposureFrom(p : vec2<f32>, travel : f32) -> f32 {
  var e = 0.0;
  for (var k = -4; k <= 4; k++) {
    let a = travel + PI + f32(k) * 0.15;
    e += (1.0 - 0.12 * abs(f32(k))) * smoothstep(150.0, 2200.0, openFetch(p, vec2<f32>(cos(a), sin(a))));
  }
  return e / 6.6;
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
      let tl = treeLine(q, terrainNormal(q, 4.0), auxAt(q), 0.0);   // the canopy slopes down to the beach (the shaders apply the same factor)
      sh += crownH(q, cr, 1.0) * tl;
      cnt += select(0.0, 1.0, cr.H * tl > 0.5);
    }
  }
  // shore character near the waterline (land and water within ~150 m of it): exposure to the swell (or, less, the wind sea) and rockiness, the
  // coast's steepness over ~10 m (a sand beach is gentle; headland tips and islet flanks are steep rock)
  let pc = org + 0.5 * cell;
  var expo = 1.0; var rocky = 0.0;
  if (abs(auxAt(pc).x) < 150.0) {
    expo = max(exposureFrom(pc, B.c.x), 0.7 * exposureFrom(pc, B.c.y));
    rocky = coastSteep(pc);
  }
  textureStore(outTex, vec2<i32>(id.xy), vec4<f32>(sh / 16.0 / B.b.z, cnt / 16.0, expo, rocky));
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
        { binding: 1, visibility: C, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float' } }] });
      const mod = await CV.shader(gpu, 'canopyBake', bakeCode());
      const lay = dev.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.l1] });
      this.pCells = dev.createComputePipeline({ label: 'bakeCells', layout: lay, compute: { module: mod, entryPoint: 'bakeCells' } });
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
        { binding: 11, resource: res.noise.createView() }, { binding: 12, resource: crownView }, { binding: 13, resource: d.u8 }, { binding: 14, resource: d.u8 }] });
      const ub = (a, b, c) => { const buf = CV.buffer(gpu, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'bakeU'); gpu.queue.writeBuffer(buf, 0, new Float32Array([...a, ...b, ...c])); return buf; };
      const lattice = (t.site && t.site.matSet === 1) ? 7.0 : this.lattice;   // closed mangrove canopy needs a denser crown lattice
      // lattice cells covering the crown window plus a margin of two cells (a texel looks at its cell's 3x3 neighbourhood)
      const ci = Math.floor(cr.x0 / lattice) - 2, cj = Math.floor(cr.z0 / lattice) - 2;
      const cnx = Math.ceil((cr.x0 + cr.nx * cr.texel) / lattice) + 2 - ci, cnz = Math.ceil((cr.z0 + cr.nz * cr.texel) / lattice) + 2 - cj;
      const cellTex = dev.createTexture({ label: 'crownCells', size: [cnx, cnz], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
      const cells = [ci, cj, cnx, cnz];
      const u0 = ub([cr.x0, cr.z0, cr.texel, cr.nx], [cr.nz, lattice, cr.maxH, 0], cells);
      const site = t.site || {};
      const u2 = ub([n.x0, n.z0, n.dx, n.nx], [n.nz, lattice, cr.maxH, 0],
        [CV.Waves.travelAngle(site.swellFrom || 165), CV.Waves.travelAngle(site.windFrom || 70), 0, 0]);
      const g1 = (u, tex, readView) => dev.createBindGroup({ layout: this.l1, entries: [{ binding: 0, resource: { buffer: u } }, { binding: 1, resource: tex.createView() }, { binding: 2, resource: readView }] });
      const enc = dev.createCommandEncoder({ label: 'canopyBake' });
      let p = enc.beginComputePass({ label: 'bakeCells' });
      p.setPipeline(this.pCells); p.setBindGroup(0, sceneBG(d.u8)); p.setBindGroup(1, g1(u0, cellTex, d.u8));
      p.dispatchWorkgroups(Math.ceil(cnx / 8), Math.ceil(cnz / 8), 1); p.end();
      p = enc.beginComputePass({ label: 'bakeCrowns' });
      p.setPipeline(this.pCrowns); p.setBindGroup(0, sceneBG(d.u8)); p.setBindGroup(1, g1(u0, t.crownTex, cellTex.createView()));
      p.dispatchWorkgroups(Math.ceil(cr.nx / 8), Math.ceil(cr.nz / 8), 1); p.end();
      p = enc.beginComputePass({ label: 'bakeMean' });
      p.setPipeline(this.pMean); p.setBindGroup(0, sceneBG(t.crownTex.createView())); p.setBindGroup(1, g1(u2, t.meanTex, d.u8));
      p.dispatchWorkgroups(Math.ceil(n.nx / 8), Math.ceil(n.nz / 8), 1); p.end();
      dev.queue.submit([enc.finish()]);
      // the per-cell crowns stay: the 3D trees near the camera are placed from them (CV.Trees), one per lattice cell that has a tree
      t.cells = { tex: cellTex, ci, cj, nx: cnx, nz: cnz, lattice };
      t.canopyBaked = true;
    }
  };
})();
