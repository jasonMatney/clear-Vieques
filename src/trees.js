// ClearVieques — 3D trees near the camera. Every crown the canopy bake placed (terrain.cells: one crown per lattice cell that has a tree) and
// whose centre is within ~90 m of the camera is drawn as a tree instead of a dome on the terrain: a tapered trunk, four limbs and ~20 leaf clusters.
// A compute pass lists the crowns in a square window of lattice cells around the camera each frame (centre, height after the tree-line taper,
// the same elliptical shape the dome has, leaf colour from the land classes there); the vertex shaders build each tree from that record and a
// per-tree hash, with no vertex buffers. Leaf clusters are camera-facing cards cut out of the leaf-cluster texture (src/foliage.js), lit with a
// mix of the crown's ellipsoid normal and a rounded card normal, self-shadowed by the depth of crown the sun ray crosses, translucent when
// backlit, swaying in the wind. Between tree3DW's fade radii a tree dissolves (dithered, averaged out by TAA) as its dome rises on the terrain.
(function () {
  'use strict';
  const CV = window.CV;
  const NC = 20;           // leaf clusters per tree
  const NSEG = 5;          // trunk + 4 limbs
  const FADE = [60, 88];   // m from the camera to the crown centre: 3D inside the first, dome beyond the second

  const treeStruct = /* wgsl */`
// a: centre x, z, ground height, crown top above the ground (0 = no tree) ; b: crown radius along rot, across it, rot (cos, sin)
// c: leaf albedo, weight as a 3D tree (tree3DW) ; d: id, mangrove, scrub (sea grape), dry-season mix
struct Tree { a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32> };
`;

  const computeCode = () => CV.wgsl.prelude() + CV.wgsl.materials + treeStruct + /* wgsl */`
struct TU { a : vec4<f32>, b : vec4<f32>, c : vec4<f32> };   // a: window's first cell i, j, cells per side, lattice (m) ; b: cell texture's first cell i, j, size ; c: max crown height
@group(1) @binding(0) var<uniform> TP : TU;
@group(1) @binding(1) var cells : texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> trees : array<Tree>;

@compute @workgroup_size(8, 8, 1)
fn place(@builtin(global_invocation_id) id : vec3<u32>) {
  let n = u32(TP.a.z);
  if (id.x >= n || id.y >= n) { return; }
  var t : Tree; t.a = vec4<f32>(0.0); t.b = vec4<f32>(0.0); t.c = vec4<f32>(0.0); t.d = vec4<f32>(0.0);
  let g = TP.a.xy + vec2<f32>(id.xy);
  let ti = vec2<i32>(g - TP.b.xy);
  if (ti.x >= 0 && ti.y >= 0 && ti.x < i32(TP.b.z) && ti.y < i32(TP.b.w)) {
    var H = textureLoad(cells, ti, 0).r * TP.c.x;
    let c = (g + 0.2 + 0.6 * hash22(g)) * TP.a.w;                 // the bake's crown centre and id for this cell (canopy.js cellCentre / cellId)
    let cid = hash21(g + vec2<f32>(3.7, 1.3));
    let w3 = tree3DW(c);
    if (H > 0.5 && w3 > 0.0) {
      let nG = terrainNormal(c, 4.0); let aux = auxAt(c);
      H *= treeLine(c, nG, aux, 0.0);
      if (H > 0.5) {
        let cls = landClass(c, nG, aux, 0.0);
        // the dome's leaf colour (landMaterial), without its per-pixel texture
        let dryK = smoothstep(0.30, 0.95, fract(cid * 5.17 + 0.31)) * (0.30 + 0.70 * cls.dryMix) * (1.0 - cls.mangW);
        var leaf = mix(vec3<f32>(0.046, 0.066, 0.026), vec3<f32>(0.098, 0.083, 0.050), dryK);
        leaf = mix(leaf, vec3<f32>(0.018, 0.029, 0.014), smoothstep(0.55, 0.0, fract(cid * 3.77)) * 0.5);
        leaf = mix(leaf, vec3<f32>(0.056, 0.090, 0.030), cls.scrub * 0.85);
        leaf = mix(leaf, vec3<f32>(0.026, 0.046, 0.021) * (0.85 + 0.3 * fract(cid * 9.7)), cls.mangW);
        leaf *= 0.85 + 0.30 * fract(cid * 11.13);
        let cs = crownShape(c, H, cid);
        t.a = vec4<f32>(c.x, c.y, heightAt(c), H);
        t.b = vec4<f32>(1.0 / cs.iR.x, 1.0 / cs.iR.y, cs.rot.x, cs.rot.y);
        t.c = vec4<f32>(leaf, w3);
        t.d = vec4<f32>(cid, cls.mangW, cls.scrub, cls.dryMix);
      }
    }
  }
  trees[id.y * n + id.x] = t;
}
`;

  const renderCode = () => CV.wgsl.prelude() + CV.wgsl.materials + treeStruct + /* wgsl */`
@group(1) @binding(0) var<storage, read> trees : array<Tree>;
const NC : u32 = ${NC}u;

struct TreeGeo { base : vec3<f32>, H : f32, R : vec2<f32>, rot : vec2<f32>, hb : f32, sv : f32, cc : vec3<f32>, id : f32, mang : f32, scrub : f32 };
fn treeGeo(t : Tree) -> TreeGeo {
  var g : TreeGeo;
  g.base = vec3<f32>(t.a.x, t.a.z, t.a.y); g.H = t.a.w; g.R = t.b.xy; g.rot = t.b.zw; g.id = t.d.x; g.mang = t.d.y; g.scrub = t.d.z;
  var hbf = 0.34 + 0.18 * fract(g.id * 4.7);                        // bole height: dry-forest trees fork at a third to a half of their height
  hbf = mix(hbf, 0.10, g.scrub);                                     // sea grape: a shrub, leafy almost to the ground
  hbf = mix(hbf, 0.22, g.mang);
  g.hb = g.H * hbf;
  g.sv = 0.5 * (g.H - g.hb);                                         // crown: an ellipsoid from the bole's top to the crown top (flat, umbrella-like)
  g.cc = g.base + vec3<f32>(0.0, g.hb + g.sv, 0.0);
  return g;
}
// crown-local (along rot, up, across rot) -> world offset: the same axes as the dome (crownRad)
fn crownToWorld(g : TreeGeo, l : vec3<f32>) -> vec3<f32> {
  return vec3<f32>(l.x * g.rot.x - l.z * g.rot.y, l.y, l.x * g.rot.y + l.z * g.rot.x);
}
fn worldToCrown(g : TreeGeo, q : vec3<f32>) -> vec3<f32> {
  return vec3<f32>(q.x * g.rot.x + q.z * g.rot.y, q.y, -q.x * g.rot.y + q.z * g.rot.x);
}
// leaf cluster k: (world centre, card radius). Spherical-Fibonacci directions over the upper ~3/4 of the crown, each pulled in by a random amount.
fn clusterPos(g : TreeGeo, k : u32) -> vec4<f32> {
  let fk = f32(k);
  let h = hash22(vec2<f32>(g.id * 97.0 + fk * 1.7, fk * 3.1));
  let z = 1.0 - (fk + 0.5) / f32(NC) * 1.55;
  let a = fk * 2.39996 + g.id * 6.2832 + h.x * 0.6;
  let s = sqrt(max(1.0 - z * z, 0.0));
  let rf = 0.45 + 0.30 * h.y;
  let l = vec3<f32>(cos(a) * s * g.R.x * rf, z * g.sv * 0.55, sin(a) * s * g.R.y * rf);
  let rad = (0.30 + 0.12 * h.x) * 0.5 * (g.R.x + g.R.y) + 0.25 * g.sv;
  return vec4<f32>(g.cc + crownToWorld(g, l), rad);
}
fn treeSway(g : TreeGeo, P : vec3<f32>) -> vec3<f32> {
  let s = windSway(g.base.xz, nowT(), fract(g.id * 13.1)) * max(P.y - g.base.y, 0.0);
  return vec3<f32>(s.x, 0.0, s.y);
}
// Optical depth (in metres of full-density foliage) of the crown along the sun ray from P. Leaf density inside the crown ellipsoid falls from 1
// at its centre to 0 at its surface (1 - r^2), so the self-shadow fades smoothly across the crown: a solid ellipsoid would put a hard terminator
// line across every tree. The ellipsoid is a little larger than the clusters' shell, which they reach past.
fn crownPath(g : TreeGeo, P : vec3<f32>, L : vec3<f32>) -> f32 {
  let ax = vec3<f32>(g.R.x * 1.25, g.sv * 1.8, g.R.y * 1.25);
  let o = worldToCrown(g, P - g.cc) / ax; let d = worldToCrown(g, L) / ax;
  let a = dot(d, d); let b = dot(o, d); let c = dot(o, o) - 1.0;
  let disc = b * b - a * c;
  if (disc <= 0.0) { return 0.0; }
  let sq = sqrt(disc);
  let t2 = max((-b + sq) / a, 0.0); let t1 = max((-b - sq) / a, 0.0);
  if (t2 <= t1) { return 0.0; }
  // integral of (1 - |o + d t|^2) dt = -c t - b t^2 - a t^3 / 3   (t in world metres: L is a unit vector)
  let F2 = -c * t2 - b * t2 * t2 - a * t2 * t2 * t2 / 3.0;
  let F1 = -c * t1 - b * t1 * t1 - a * t1 * t1 * t1 / 3.0;
  return max(F2 - F1, 0.0);
}
const HIDDEN : vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0);              // every vertex of an absent tree lands here: zero-area, clipped

// ---------------------------------------------------------------------------------------------------------------- bark: trunk and limbs
struct BOut { @builtin(position) pos : vec4<f32>, @location(0) wpos : vec3<f32>, @location(1) n : vec3<f32>, @location(2) @interpolate(flat) iid : u32 };
@vertex fn vs_bark(@builtin(vertex_index) vid : u32, @builtin(instance_index) iid : u32) -> BOut {
  var o : BOut; o.wpos = vec3<f32>(0.0); o.n = vec3<f32>(0.0, 1.0, 0.0); o.iid = iid;
  let t = trees[iid];
  if (t.a.w <= 0.5) { o.pos = HIDDEN; return o; }
  let g = treeGeo(t);
  let s = vid / 36u; let l = vid % 36u; let side = l / 6u; let c6 = l % 6u;
  var corner = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
  let cu = corner[c6];
  let lean = (hash22(vec2<f32>(g.id * 13.0, 2.0)) - 0.5) * 0.12 * g.H;
  let F = g.base + vec3<f32>(lean.x, g.hb * 0.92, lean.y);           // where the trunk forks into limbs
  let tr = (0.07 + 0.028 * g.H) * (0.8 + 0.4 * fract(g.id * 7.3)) * mix(1.0, 0.6, g.scrub);
  var A = g.base - vec3<f32>(0.0, 0.3, 0.0); var B = F; var rA = tr * 1.25; var rB = tr;
  if (s > 0u) {
    let a = g.id * 6.2832 + f32(s) * 1.5708 + (fract(g.id * 31.7 * f32(s)) - 0.5) * 0.9;
    A = F; B = g.cc + crownToWorld(g, vec3<f32>(cos(a) * 0.58 * g.R.x, 0.15 * g.sv, sin(a) * 0.58 * g.R.y));
    rA = tr * 0.62; rB = tr * 0.22;
  }
  let ax = normalize(B - A);
  let refAx = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(ax.z) > 0.9);
  let e1 = normalize(cross(ax, refAx)); let e2 = cross(ax, e1);
  let th = (f32(side) + cu.x) / 6.0 * TAU;
  let rn = cos(th) * e1 + sin(th) * e2;
  var P = mix(A, B, cu.y) + rn * mix(rA, rB, cu.y);
  P += treeSway(g, P);
  o.wpos = P; o.n = rn;
  o.pos = G.viewProj * vec4<f32>(P, 1.0);
  return o;
}
@fragment fn fs_bark(in : BOut) -> @location(0) vec4<f32> {
  let t = trees[in.iid]; let g = treeGeo(t);
  let P = in.wpos; let toCam = G.camPos.xyz - P; let dist = length(toCam);
  let n = normalize(in.n); let L = G.sunDir.xyz;
  // grey-brown bark; some dry-forest trees are gumbo-limbo, with copper-red peeling bark; mangroves dark red-brown
  var alb = mix(vec3<f32>(0.15, 0.13, 0.11), vec3<f32>(0.27, 0.14, 0.10), step(0.75, fract(g.id * 3.3)) * (1.0 - g.mang) * (1.0 - g.scrub));
  alb = mix(alb, vec3<f32>(0.14, 0.09, 0.07), g.mang);
  alb *= 0.70 + 0.55 * vnoise4(vec2<f32>(dot(P.xz, vec2<f32>(6.0, 4.0)), P.y * 2.2)).r;
  let jit = hash21(in.pos.xy + G.frame.x * vec2<f32>(7.13, 3.71));
  let sh = sunShadow(P + n * 0.1, jit) * cloudShadow(P) * mix(0.10, 1.0, exp(-0.75 * crownPath(g, P, L)));
  // the crown overhead hides much of the sky, and the ground around the trunk is in its shade, so it bounces little light up
  let under = 0.45 + 0.25 * (1.0 - smoothstep(g.base.y, g.cc.y, P.y));
  let Eamb = (G.skyE.rgb * (0.5 + 0.5 * n.y) * 0.55 + G.sunE.rgb * max(L.y, 0.0) * vec3<f32>(0.30, 0.27, 0.22) * (0.5 - 0.5 * n.y + 0.03) * 0.45) * under;
  var col = alb / PI * (G.sunE.rgb * max(dot(n, L), 0.0) * sh + Eamb);
  col = applyFog(col, dist, -toCam / dist);
  return vec4<f32>(col, 1.0);
}

// ---------------------------------------------------------------------------------------------------------------- leaves
struct LOut {
  @builtin(position) pos : vec4<f32>, @location(0) wpos : vec3<f32>, @location(1) uv : vec2<f32>, @location(2) tuv : vec2<f32>,
  @location(3) @interpolate(flat) iid : u32, @location(4) @interpolate(flat) k : u32,
};
@vertex fn vs_leaf(@builtin(vertex_index) vid : u32, @builtin(instance_index) iid : u32) -> LOut {
  var o : LOut; o.wpos = vec3<f32>(0.0); o.uv = vec2<f32>(0.0); o.tuv = vec2<f32>(0.0); o.iid = iid; o.k = 0u;
  let t = trees[iid];
  if (t.a.w <= 0.5) { o.pos = HIDDEN; return o; }
  let g = treeGeo(t);
  let k = vid / 6u;
  var corner = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
  let uv = corner[vid % 6u];
  let cp = clusterPos(g, k);
  let ctr = cp.xyz + treeSway(g, cp.xyz);
  let P = ctr + (G.camRight.xyz * uv.x + G.camUp.xyz * uv.y) * cp.w;
  let hk = hash22(vec2<f32>(g.id * 31.0, f32(k) * 7.7));
  let a = hk.x * TAU; let ca = cos(a); let sa = sin(a);
  o.wpos = P; o.uv = uv; o.k = k;
  // each card shows its own patch of the leaf-cluster texture, at a scale where its finest clusters are a few leaves (~10 cm)
  o.tuv = vec2<f32>(ca * uv.x - sa * uv.y, sa * uv.x + ca * uv.y) * (0.50 + 0.15 * hk.y) * cp.w / 1.6 + hk * 7.0;
  o.pos = G.viewProj * vec4<f32>(P, 1.0);
  return o;
}
@fragment fn fs_leaf(in : LOut) -> @location(0) vec4<f32> {
  let tex = textureSample(foliageTex, samRep, in.tuv);              // (implicit LOD: before any discard)
  let t = trees[in.iid]; let g = treeGeo(t);
  let r2 = dot(in.uv, in.uv);
  // leafy cut-out: solid in the middle, holes and a ragged edge outward (the texture's exposure channel is high on cluster tops)
  let m = (1.0 - r2) * 1.20 + (tex.b - 0.5) * 1.7 - 0.40;
  let dither = hash21(in.pos.xy + G.frame.x * vec2<f32>(7.13, 3.71));
  if (m < 0.0 || dither > t.c.w) { discard; }                       // dissolve into the dome over the fade band (TAA averages the dither)
  let P = in.wpos; let toCam = G.camPos.xyz - P; let dist = length(toCam); let V = toCam / dist;
  let L = G.sunDir.xyz;
  // normal: the crown ellipsoid's (the whole tree reads as a lit volume) plus a rounded card normal and the cluster texture's relief
  let ax = vec3<f32>(g.R.x, g.sv, g.R.y);
  let e = worldToCrown(g, P - g.cc) / ax;
  // (the direction from the crown centre in the crown's normalised frame: a flat ellipsoid's true gradient flips from up to down within a few
  // centimetres of its equator, which drew a hard light/dark line across every crown)
  let nC = normalize(crownToWorld(g, vec3<f32>(e.x, e.y * 0.8, e.z)));
  let nS = normalize(G.camRight.xyz * in.uv.x + G.camUp.xyz * in.uv.y - G.camFwd.xyz * sqrt(max(1.0 - r2, 0.0)));
  let tp = (tex.rg - 0.5) * 1.4;
  let n = normalize(nC * 0.80 + nS * 0.45 + G.camRight.xyz * tp.x + G.camUp.xyz * tp.y);
  // albedo: the tree's leaf colour, cluster to cluster variation, fresh yellow-green flush on a few clusters
  let hk = hash21(vec2<f32>(g.id * 31.0, f32(in.k) * 7.7 + 1.3));
  var alb = t.c.rgb * (0.70 + 0.36 * hk) * (1.0 + 0.45 * (tex.a - 0.5));
  alb = mix(alb, alb * vec3<f32>(1.30, 1.22, 0.62), step(0.82, hk) * (1.0 - g.mang) * 0.6);
  // light: sun through the crown in front of this leaf, the terrain and the stand, the clouds; deeper and lower leaves see less sky
  let jit = hash21(in.pos.xy + G.frame.x * vec2<f32>(3.1, 9.7));
  let trans = mix(0.15, 1.0, exp(-0.75 * crownPath(g, P, L)));       // foliage scatters some light through the crown
  let sh = sunShadow(P + L * 0.4, jit) * cloudShadow(P) * trans;
  let depth = length(e);
  let ao = mix(0.30, 1.0, smoothstep(0.15, 0.95, depth)) * (0.70 + 0.30 * smoothstep(-0.8, 0.8, e.y)) * (0.45 + 0.75 * tex.b);   // clefts between clusters are dark
  let wrap = 0.35;
  let ndl = max((dot(n, L) + wrap) / (1.0 + wrap), 0.0);
  let Eamb = G.skyE.rgb * (0.55 + 0.45 * n.y) * ao + G.sunE.rgb * max(L.y, 0.0) * vec3<f32>(0.30, 0.27, 0.22) * (0.5 - 0.5 * n.y + 0.03) * ao;
  var col = alb / PI * (G.sunE.rgb * ndl * sh * (0.55 + 0.45 * tex.b) + Eamb);
  // leaves transmit: looking toward the sun, sunlit leaves glow yellow-green; a waxy sheen on clusters facing the half vector
  col += alb * vec3<f32>(1.0, 1.25, 0.45) * G.sunE.rgb / PI * (0.6 * pow(max(dot(-V, L), 0.0), 2.0) * sh);
  let nh = max(dot(n, normalize(L + V)), 0.0);
  col += G.sunE.rgb * (0.012 * 26.0 / (8.0 * PI) * pow(nh, 18.0) * ndl * sh);
  col = applyFog(col, dist, -V);
  return vec4<f32>(col, 1.0);
}
`;

  CV.Trees = class Trees {
    constructor(gpu) { this.gpu = gpu; this.fade = FADE; this.n = 0; }

    async init(sceneLayout) {
      const dev = this.gpu.device, C = GPUShaderStage.COMPUTE, VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
      this.sceneLayout = sceneLayout;
      this.cBGL = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, texture: { sampleType: 'float' } },
        { binding: 2, visibility: C, buffer: { type: 'storage' } }] });
      this.rBGL = dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: { type: 'read-only-storage' } }] });
      const [mC, mR] = await Promise.all([CV.shader(this.gpu, 'treesPlace', computeCode()), CV.shader(this.gpu, 'trees', renderCode())]);
      this.mR = mR;
      this.pPlace = dev.createComputePipeline({ label: 'treesPlace', layout: dev.createPipelineLayout({ bindGroupLayouts: [sceneLayout, this.cBGL] }),
        compute: { module: mC, entryPoint: 'place' } });
      this.ubuf = CV.buffer(this.gpu, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'treesU');
    }

    buildPipelines(sampleCount) {
      const dev = this.gpu.device, layout = dev.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.rBGL] });
      const mk = (vs, fs, label) => dev.createRenderPipeline({ label, layout, vertex: { module: this.mR, entryPoint: vs },
        fragment: { module: this.mR, entryPoint: fs, targets: [{ format: 'rgba16float' }] }, multisample: { count: sampleCount },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: { format: 'depth32float', depthCompare: 'greater', depthWriteEnabled: true } });
      this.pBark = mk('vs_bark', 'fs_bark', 'treeBark');
      this.pLeaf = mk('vs_leaf', 'fs_leaf', 'treeLeaves');
    }

    // (re)size the per-tree buffer for a terrain's crown lattice: a square window of cells that covers the fade radius around the camera
    bindTerrain(terrain) {
      const c = terrain.cells, dev = this.gpu.device;
      this.terrain = terrain;
      this.n = Math.ceil(2 * this.fade[1] / c.lattice) + 3;
      if (this.buf) this.buf.destroy();
      this.buf = dev.createBuffer({ label: 'trees', size: this.n * this.n * 64, usage: GPUBufferUsage.STORAGE });
      this.cBG = dev.createBindGroup({ layout: this.cBGL, entries: [{ binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: c.tex.createView() }, { binding: 2, resource: { buffer: this.buf } }] });
      this.rBG = dev.createBindGroup({ layout: this.rBGL, entries: [{ binding: 0, resource: { buffer: this.buf } }] });
    }

    // list the crowns around the camera (the scene bind group carries the globals: camera, tree3DW's fade radii)
    encode(enc, sceneBG, terrain, camPos) {
      if (!terrain.cells) return false;
      if (this.terrain !== terrain) this.bindTerrain(terrain);
      const c = terrain.cells, lat = c.lattice;
      const i0 = Math.floor((camPos[0] - this.fade[1]) / lat) - 1, j0 = Math.floor((camPos[2] - this.fade[1]) / lat) - 1;
      this.gpu.queue.writeBuffer(this.ubuf, 0, new Float32Array([i0, j0, this.n, lat, c.ci, c.cj, c.nx, c.nz, terrain.crown.maxH, 0, 0, 0]));
      const p = enc.beginComputePass({ label: 'treesPlace', timestampWrites: CV.tw('trees') });
      p.setPipeline(this.pPlace); p.setBindGroup(0, sceneBG); p.setBindGroup(1, this.cBG);
      p.dispatchWorkgroups(Math.ceil(this.n / 8), Math.ceil(this.n / 8), 1); p.end();
      return true;
    }

    draw(pass) {
      const count = this.n * this.n;
      pass.setBindGroup(1, this.rBG);
      pass.setPipeline(this.pBark); pass.draw(NSEG * 36, count);
      pass.setPipeline(this.pLeaf); pass.draw(NC * 6, count);
    }
  };
})();
