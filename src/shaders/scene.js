// ClearVieques — render shaders: sky, terrain (land + wet sand), water surface (reflection / refraction / absorption / caustics / foam), post.
(function () {
  'use strict';
  const CV = window.CV;
  CV.wgsl = CV.wgsl || {};

  // ------------------------------------------------------------------------------------------------ materials (shared by terrain + reflections + seabed)
  const materials = /* wgsl */`
fn fadeN(n : vec4<f32>, dist : f32, d0 : f32, d1 : f32) -> vec4<f32> { return mix(n, vec4<f32>(0.5), smoothstep(d0, d1, dist)); }

fn hash22(p : vec2<f32>) -> vec2<f32> { return vec2<f32>(hash21(p), hash21(p + vec2<f32>(19.19, 7.77))); }

// Cellular "crowns": x = distance to the nearest crown centre (crown units), y = random id, zw = vector to the centre.
fn crownField(p : vec2<f32>, size : f32) -> vec4<f32> {
  let q = p / size; let i = floor(q); let f = fract(q);
  var best = 9.0; var id = 0.0; var vc = vec2<f32>(0.0);
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    let g = vec2<f32>(f32(x), f32(y));
    let o = g + 0.2 + 0.6 * hash22(i + g) - f;
    let d = dot(o, o);
    if (d < best) { best = d; id = hash21(i + g + vec2<f32>(3.7, 1.3)); vc = o; }
  } }
  return vec4<f32>(sqrt(best), id, vc);
}

// ---- foliage micro-structure. A canopy is not a smooth dome: it is leaf clusters piled in clusters of clusters, lit on top and dark between.
// CV.makeFoliageTexture bakes that structure (see src/foliage.js); three octaves of it (tile 20 m, 6.4 m, 2 m: clusters of ~3 m, ~1 m, ~0.3 m with finer
// structure inside) give each crown pixel a surface slope (g, d height / d position), an occlusion term (ao: low between clusters, mean 1) and a leaf
// tint. The explicit mip level follows the pixel footprint, so detail finer than a pixel averages out (no shimmer, no brightness change at range).
struct Foliage { ao : f32, g : vec2<f32>, tone : f32 };
fn foliageOct(p : vec2<f32>, tile : f32, cs : vec2<f32>, off : vec2<f32>, foot : f32, slope : f32) -> vec4<f32> {   // (ao, gx, gz, tone)
  let lod = log2(max(foot * 256.0 / tile, 1e-3)) - 0.4;
  if (lod > 6.5) { return vec4<f32>(1.0, 0.0, 0.0, 0.0); }
  let q = vec2<f32>(cs.x * p.x - cs.y * p.y, cs.y * p.x + cs.x * p.y) / tile + off;
  let t = textureSampleLevel(foliageTex, samRep, q, max(lod, 0.0));
  let g = (t.rg * 2.0 - 1.0) * slope;
  let fade = 1.0 - smoothstep(4.0, 6.5, lod);
  return vec4<f32>(mix(1.0, 0.40 + 0.90 * sqrt(t.b), fade), cs.x * g.x + cs.y * g.y, -cs.y * g.x + cs.x * g.y, (t.a - 0.5) * fade);
}
fn foliage(p : vec2<f32>, foot : f32, seed : f32) -> Foliage {
  let o = vec2<f32>(seed, seed * 0.61);
  let a = foliageOct(p, 20.0, vec2<f32>(0.9394, 0.3429), o, foot, 0.90);
  let b = foliageOct(p, 6.4, vec2<f32>(0.4536, 0.8912), o + 7.3, foot, 0.70);
  let c = foliageOct(p, 2.0, vec2<f32>(-0.4688, 0.8833), o + 3.1, foot, 0.45);
  var f : Foliage;
  f.ao = a.x * b.x * c.x;
  f.g = a.yz + b.yz + c.yz;
  f.tone = a.w * 0.9 + b.w * 0.7 + c.w * 0.5;
  return f;
}

fn sandAlbedo(p : vec2<f32>, dist : f32) -> vec3<f32> {
  let n1 = vnoise4(p * 0.028);
  let n2 = fadeN(vnoise4(p * 0.35), dist, 40.0, 300.0);
  let grain = fadeN(vnoise4(p * 8.5), dist, 8.0, 70.0);
  let speck = fadeN(vnoise4(p * 23.0), dist, 4.0, 30.0);     // shell / coral fragments and dark grains
  let white = vec3<f32>(0.50, 0.47, 0.42);
  let pink = vec3<f32>(0.49, 0.37, 0.32);                    // foraminifera-rich sand: the "red" of Red Beach
  let a = mix(white, pink, smoothstep(0.50, 0.88, n1.g) * 0.55 + 0.20 * smoothstep(0.60, 0.90, n2.r));
  var c = a * (0.91 + 0.18 * grain.g);
  c *= 1.0 + 0.24 * smoothstep(0.86, 0.97, speck.r) - 0.20 * smoothstep(0.12, 0.03, speck.g);
  return c;
}

// ---- tree line: 1 where crowns stand at full height, falling to 0 over ~6 m toward the open beach, so the canopy slopes down to the sand like
// wind-pruned sea grape instead of ending in a wall. The vertex shader (canopy lift), the fragment shader (crown shading) and the canopy-mean bake
// all apply the same factor, continuously in position (the crown map itself is 2 m texels).
fn mangroveW(aux : vec4<f32>, slope : f32, elev : f32) -> f32 {
  return (1.0 - smoothstep(70.0, 230.0, aux.w)) * (1.0 - smoothstep(1.2, 3.4, elev)) * (1.0 - smoothstep(0.10, 0.30, slope)) * step(0.5, G.misc.w);
}
fn treeLineK(sandEdge : f32, sd : f32, slope : f32, elev : f32, mangW : f32) -> f32 {
  let beach = (1.0 - smoothstep(sandEdge - 1.0, sandEdge + 5.0, sd)) * (1.0 - smoothstep(0.08, 0.24, slope)) * (1.0 - smoothstep(3.2, 5.5, elev));
  return max(1.0 - beach, mangW);
}
fn treeLine(p : vec2<f32>, n : vec3<f32>, aux : vec4<f32>, dist : f32) -> f32 {
  let n1 = vnoise4(p * 0.018); let n3 = fadeN(vnoise4(p * 0.75), dist, 60.0, 420.0);
  let elev = heightAt(p) - seaLevel(); let slope = 1.0 - n.y;
  return treeLineK(9.0 + 26.0 * n1.g + 7.0 * (n3.r - 0.5), aux.x, slope, elev, mangroveW(aux, slope, elev));
}

// ---- land classification: shared by the canopy bake (compute, dist = 0) and the terrain shading (fragment)
struct LandCls {
  sandW : f32, rockW : f32, plain : f32, scrub : f32, trackW : f32, vegW : f32, dryMix : f32, sandEdge : f32, elev : f32, mangW : f32, sd : f32, treeLine : f32,
  n1 : vec4<f32>, n2 : vec4<f32>, n3 : vec4<f32>, n4 : vec4<f32>,
};
fn landClass(pxz : vec2<f32>, n : vec3<f32>, aux : vec4<f32>, dist : f32) -> LandCls {
  var c : LandCls;
  c.elev = heightAt(pxz) - seaLevel();
  let sd = aux.x;
  c.n1 = vnoise4(pxz * 0.018);
  c.n2 = fadeN(vnoise4(pxz * 0.11), dist, 900.0, 3500.0);
  c.n3 = fadeN(vnoise4(pxz * 0.75), dist, 60.0, 420.0);
  c.n4 = fadeN(vnoise4(pxz * 3.1), dist, 25.0, 160.0);
  // slope with a little noise: the DEM is bilinear on 4 m cells, so a bare slope threshold draws the sand / rock / scrub boundaries as straight
  // polygon edges along its facets
  let slope = 1.0 - n.y + 0.07 * (c.n3.g - 0.5) + 0.04 * (c.n4.b - 0.5) + 0.05 * (c.n2.r - 0.5);
  // beach: back-beach limit varies along the shore; the low coastal plain behind it is dry scrub / grass, not sand
  c.sandEdge = 9.0 + 26.0 * c.n1.g + 7.0 * (c.n3.r - 0.5);          // irregular vegetation line
  var sandW = 1.0 - smoothstep(c.sandEdge, c.sandEdge + 1.6, sd);
  sandW *= 1.0 - smoothstep(0.08, 0.24, slope);
  sandW *= 1.0 - smoothstep(3.2, 5.5, c.elev);
  c.sandW = sandW;
  c.rockW = smoothstep(0.17, 0.34, slope) * (0.55 + 0.45 * c.n2.b) * (1.0 - sandW);
  c.dryMix = smoothstep(0.35, 0.80, c.n1.r * 0.55 + c.n2.g * 0.45);
  c.plain = (1.0 - smoothstep(1.5, 6.0, c.elev)) * (1.0 - smoothstep(0.03, 0.10, slope));       // low, flat coastal plain: grass and scrub
  c.scrub = smoothstep(c.sandEdge - 1.0, c.sandEdge + 3.0, sd) * (1.0 - smoothstep(c.sandEdge + 6.0, c.sandEdge + 40.0, sd));   // sea-grape band behind the sand
  c.trackW = 1.0 - smoothstep(1.1, 2.3, aux.z);
  // mangrove fringe (lagoon sites): low, gentle ground within ~100-200 m of the lagoon's water (aux.w = distance to it)
  c.mangW = mangroveW(aux, slope, c.elev);
  c.sd = sd;
  c.treeLine = treeLineK(c.sandEdge, sd, slope, c.elev, c.mangW);
  c.sandW *= 1.0 - c.mangW; c.plain *= 1.0 - c.mangW; c.scrub *= 1.0 - c.mangW;
  c.vegW = (1.0 - c.sandW) * (1.0 - c.rockW) * (1.0 - c.trackW);
  return c;
}

var<private> gHole : f32 = 0.0;        // set by shadeLand: 1 where the leafy fringe of a crown lets open sky show through (the terrain pass paints the sky there; no discard, which would defeat tile-based hidden-surface removal)
struct LandMat { albedo : vec3<f32>, sandW : f32, rockW : f32, canopy : f32, nAdd : vec3<f32>, occ : f32, leaf : f32, hTop : f32, rim : f32 };

fn landMaterial(pxz : vec2<f32>, n : vec3<f32>, V : vec3<f32>, dist : f32, c : LandCls) -> LandMat {
  // dry-forest palette (linear reflectance), a little muted: this is a tropical dry forest in the trades, not a rainforest
  let leafGreen = vec3<f32>(0.046, 0.066, 0.026);
  let leafDark = vec3<f32>(0.018, 0.029, 0.014);
  let leafDry = vec3<f32>(0.098, 0.083, 0.050);
  let seaGrape = vec3<f32>(0.056, 0.090, 0.030);
  let grass = vec3<f32>(0.150, 0.135, 0.070);
  let under = vec3<f32>(0.034, 0.025, 0.016);

  var cr = crownAt(pxz);
  cr.H *= c.treeLine;                                                // the canopy slopes down to the beach (same factor as the vertex lift)
  let treeFrac = meanCanopy(pxz).y;
  // metres covered by one pixel on the foliage: individual crowns (~3-5 m radius) are drawn only where they are resolved, otherwise the stand mean.
  // Crowns bulge toward the viewer, so unlike bare ground their footprint does not stretch at grazing angles (the cap on 1/cos is mild).
  let foot = dist * G.camFwd.w / max(dot(n, V), 0.45);
  let resolved = 1.0 - smoothstep(1.8, 5.5, foot);
  let hasCrown = cr.H > 0.5;
  let rr = crownRad(pxz, cr);
  let inCrown = select(0.0, 1.0 - smoothstep(0.92, 1.12, rr), hasCrown);
  let cover = mix(treeFrac, inCrown, resolved);                     // how much of this pixel is foliage
  let canopyMean = mix(leafGreen, leafDry, c.dryMix * 0.55) * 0.80;
  // dry grass and litter between trees and on the plain, with tufts (~10-30 cm) and bare-soil flecks visible at close range
  let g1 = fadeN(vnoise4(pxz * 9.0), dist, 10.0, 60.0); let g2 = fadeN(vnoise4(pxz * 31.0), dist, 4.0, 22.0);
  var groundCover = mix(grass, leafDry, 0.35 * c.n3.g) * (0.75 + 0.5 * c.n4.r) * (0.80 + 0.40 * g1.r) * (0.88 + 0.24 * g2.g);
  groundCover = mix(groundCover, vec3<f32>(0.060, 0.082, 0.032), smoothstep(0.62, 0.86, g1.g) * 0.55);          // green tufts
  groundCover = mix(groundCover, vec3<f32>(0.19, 0.15, 0.10), smoothstep(0.80, 0.95, g2.r) * 0.35);            // bare soil flecks
  groundCover = mix(groundCover, vec3<f32>(0.052, 0.042, 0.030) * (0.7 + 0.6 * g1.g), c.mangW);                 // mangrove mud
  let meanCol = mix(groundCover, canopyMean, treeFrac);
  var veg = meanCol; var occ = 1.0; var nAdd = vec3<f32>(0.0); var leafW = 0.0; var hTop = 0.0; var rimW = 0.0;
  if (resolved > 0.0) {
    var detailCol = groundCover;
    if (hasCrown) {
      let idv = cr.id;
      let hfrac = crownProfile(rr, treeFrac);
      let dryK = smoothstep(0.30, 0.95, fract(idv * 5.17 + 0.31)) * (0.30 + 0.70 * c.dryMix) * (1.0 - c.mangW);      // deciduous crowns go yellow-brown in the dry season
      var leaf = mix(leafGreen, leafDry, dryK);
      leaf = mix(leaf, leafDark, smoothstep(0.55, 0.0, fract(idv * 3.77)) * 0.5);                    // dark evergreen crowns
      leaf = mix(leaf, seaGrape, c.scrub * 0.85);
      leaf = mix(leaf, vec3<f32>(0.026, 0.046, 0.021) * (0.85 + 0.3 * fract(idv * 9.7)), c.mangW);    // red mangrove: dense, dark, glossy
      leaf *= 0.85 + 0.30 * fract(idv * 11.13);
      let ln = fadeN(vnoise4(pxz * 1.9 + vec2<f32>(idv * 17.0, 3.0)), dist, 30.0, 220.0);            // leaf clumps (~0.5 m)
      let ls = fadeN(vnoise4(pxz * 7.3 + vec2<f32>(5.0, idv * 9.0)), dist, 8.0, 70.0);               // leaf speckle (~15 cm)
      leaf *= (0.78 + 0.44 * ln.r) * (0.86 + 0.28 * ls.g);
      // leaf-cluster structure: every cluster has its own tint (fresh yellow-green flush, blue-green shade leaves) and the pile has creases and relief
      let fol = foliage(pxz, foot, idv * 41.0);
      leaf *= 1.0 + 0.50 * fol.tone;
      leaf = mix(leaf, leaf * vec3<f32>(1.30, 1.22, 0.62), smoothstep(0.10, 0.45, fol.tone) * (1.0 - c.mangW) * 0.5);
      leaf = mix(leaf, leaf * vec3<f32>(0.72, 0.95, 1.18), smoothstep(-0.12, -0.50, fol.tone) * 0.6);
      // between crowns: shaded foliage inside a dense stand (dark green), bare litter beyond it
      let floorFol = smoothstep(0.55, 0.90, treeFrac) * (1.0 - smoothstep(1.30, 2.10, rr));
      // leaf litter lies under the crown and just past its drip line; further out it is the open ground cover (a texel's crown can be one whose
      // canopy is metres away, and its region ends on straight Voronoi edges, so the litter must follow the crown radius, not the texel)
      let litter = 1.0 - smoothstep(1.0, 1.6, rr);
      detailCol = mix(mix(mix(groundCover, under * (0.7 + 0.6 * ln.g), litter), leafDark * 0.75 * (0.8 + 0.4 * ln.g), floorFol), leaf, inCrown);
      // light and shade come from the dome itself: tilted normal, dark low gaps, bright tops. Gaps between crowns are only visible when
      // looking down into the canopy; at grazing angles the crowns in front hide them (this removes the dark rim along ridges).
      let gapVis = smoothstep(0.10, 0.55, clamp(dot(n, V), 0.0, 1.0));
      let underForest = 1.0 - c.sandW;                       // open sand beside a crown is not understorey shade
      let folK = max(inCrown, 0.75 * floorFol) * resolved * underForest;
      // (only under foliage or inside a dense stand: open ground beside a crown is not in its gap shade, and the texels a crown owns end on
      // straight Voronoi edges that would show as dark polygons)
      occ = mix(1.0, mix(0.28, 1.0, smoothstep(0.52, 0.92, hfrac)), gapVis * resolved * underForest * max(inCrown, floorFol));
      occ *= mix(1.0, clamp(fol.ao, 0.22, 1.45), folK);
      let g = crownGrad(pxz, cr, treeFrac);
      nAdd = vec3<f32>(-(g.x + fol.g.x * folK), 0.0, -(g.y + fol.g.y * folK)) * 0.85 * resolved * underForest;
      leafW = max(inCrown, 0.5 * floorFol) * resolved;
      hTop = crownH(pxz, cr, treeFrac);
      rimW = inCrown * resolved * smoothstep(0.50, 0.85, hfrac);     // only the upper dome frays at the edge; the lower flanks stay solid
    }
    veg = mix(meanCol, detailCol, resolved);
  }
  let rock = mix(vec3<f32>(0.31, 0.28, 0.23), vec3<f32>(0.19, 0.17, 0.14), c.n3.g) * (0.70 + 0.50 * c.n4.r);
  let soil = vec3<f32>(0.30, 0.22, 0.14) * (0.85 + 0.3 * c.n4.b);
  // beach morning-glory and grass runners creep a few metres onto the sand from the vegetation line, in patches
  let creepBand = smoothstep(c.sandEdge - 7.0, c.sandEdge - 0.5, c.sd) * c.sandW * (1.0 - c.mangW);
  let cn = fadeN(vnoise4(pxz * 0.45 + vec2<f32>(3.7, 1.1)), dist, 80.0, 400.0); let cf = fadeN(vnoise4(pxz * 3.2), dist, 15.0, 90.0);
  let creep = smoothstep(0.60, 0.80, cn.r + 0.40 * creepBand - 0.22) * creepBand * (0.55 + 0.45 * cf.g) * (1.0 - cover);
  let sandW = c.sandW * (1.0 - 0.97 * cover) * (1.0 - 0.85 * creep);  // a tree standing on the sand line is a tree, not sand
  var alb = mix(veg, rock, c.rockW);
  // the beach is not one flat tone: broad patches of slightly darker / pinker sand, and trampled sand between the swash and the vegetation
  // where footprints leave small shadowed dimples
  var sandCol = sandAlbedo(pxz, dist) * (0.93 + 0.14 * c.n2.b);
  let tramp = smoothstep(4.0, 10.0, c.sd) * (1.0 - smoothstep(c.sandEdge - 5.0, c.sandEdge, c.sd)) * smoothstep(0.30, 0.60, c.n1.b);
  let dn = fadeN(vnoise4(pxz * 2.9 + vec2<f32>(1.3, 4.1)), dist, 6.0, 40.0);
  sandCol *= 1.0 - 0.18 * tramp * smoothstep(0.56, 0.80, dn.r) + 0.05 * tramp * smoothstep(0.55, 0.80, dn.g);
  alb = mix(alb, sandCol, sandW);
  alb = mix(alb, vec3<f32>(0.048, 0.080, 0.028) * (0.75 + 0.5 * cf.b), creep * 0.85);
  alb = mix(alb, soil, c.trackW * (1.0 - 0.6 * sandW) * (1.0 - cover));
  var m : LandMat;
  m.albedo = alb; m.sandW = sandW; m.rockW = c.rockW;
  m.canopy = clamp(cover, 0.0, 1.0) * (1.0 - c.rockW);
  m.nAdd = nAdd; m.occ = occ; m.leaf = leafW * (1.0 - c.rockW) * (1.0 - c.sandW); m.hTop = hTop; m.rim = rimW * (1.0 - c.rockW) * (1.0 - c.sandW);
  return m;
}

// Sun shadow cast onto a crown by its neighbours (the global march only sees the smoothed stand): walk toward the sun over the crown height field.
// hSelf = canopy height of the point above the bare earth; nG = bare-earth normal (a hillside rising toward the sun lifts the occluders).
fn crownShade(pxz : vec2<f32>, nG : vec3<f32>, L : vec3<f32>, tf : f32, hSelf : f32, jit : f32) -> f32 {
  let lh = length(L.xz);
  if (lh < 0.02) { return 1.0; }
  let dir = L.xz / lh;
  let tanE = L.y / lh;                                              // rise of the sun ray per metre walked toward the sun
  let slope = -nG.xz / max(nG.y, 0.2);                              // bare-earth rise per metre (d h / d x, d h / d z)
  let dmax = clamp(7.0 / max(tanE, 0.05), 6.0, 18.0);
  var vis = 1.0;
  for (var k = 0; k < 5; k++) {
    let d = dmax * ((f32(k) + 0.5 + 0.5 * (jit - 0.5)) / 5.0);
    let q = pxz + dir * d;
    let occ = crownH(q, crownAt(q), tf) + clamp(dot(slope, dir) * d, -4.0, 4.0);
    vis = min(vis, smoothstep(-0.6, 0.7, hSelf + d * tanE - occ));
  }
  return vis;
}

fn seabedAlbedo(q : vec3<f32>, nb : vec3<f32>, depth : f32, dist : f32) -> vec3<f32> {
  let p = q.xz;
  let n1 = vnoise4(p * 0.030);
  let n2 = fadeN(vnoise4(p * 0.19), dist, 200.0, 900.0);
  let n3 = fadeN(vnoise4(p * 1.30), dist, 25.0, 160.0);
  let n4 = fadeN(vnoise4(p * 5.20), dist, 10.0, 70.0);
  if (G.misc.w > 0.5) {                                        // lagoon floor: dark mud with seagrass, not bright sand
    let mud = vec3<f32>(0.120, 0.098, 0.072) * (0.90 + 0.20 * n3.g);
    let sg = vec3<f32>(0.030, 0.052, 0.024) * (0.75 + 0.5 * n3.r);
    return mix(mud, sg, smoothstep(0.42, 0.60, n1.r * 0.62 + n2.g * 0.38) * smoothstep(0.25, 1.1, depth));
  }
  let sandW = vec3<f32>(0.68, 0.62, 0.52);
  let sandP = vec3<f32>(0.62, 0.49, 0.40);
  var sand = mix(sandW, sandP, smoothstep(0.38, 0.75, n1.b));
  sand *= 0.90 + 0.20 * n4.g;
  // wave-formed ripple marks (aligned with the swell), ~14 cm crest spacing
  let rip = sin(dot(p, vec2<f32>(0.62, -0.78)) * 44.0 + n2.r * 7.0);
  sand *= 1.0 + 0.07 * rip * smoothstep(0.2, 1.2, depth) * (1.0 - smoothstep(30.0, 120.0, dist));
  // seagrass beds
  let gm = smoothstep(0.50, 0.64, n1.r * 0.62 + n2.g * 0.38) * smoothstep(1.4, 2.6, depth) * (1.0 - smoothstep(9.0, 15.0, depth));
  let grass = vec3<f32>(0.030, 0.052, 0.024) * (0.55 + 0.9 * n3.r);
  // hard-ground / rubble on slopes and near rocky shores
  let steep = 1.0 - nb.y;
  let rockW = smoothstep(0.10, 0.26, steep) * (0.6 + 0.4 * n2.b);
  let rock = vec3<f32>(0.13, 0.115, 0.09) * (0.6 + 0.8 * n3.g);
  var a = mix(sand, grass, gm);
  a = mix(a, rock, rockW);
  return a;
}
`;

  CV.wgsl.materials = materials;

  // ------------------------------------------------------------------------------------------------ sky
  CV.wgsl.sky = /* wgsl */`
struct SkyVOut { @builtin(position) pos : vec4<f32>, @location(0) ndc : vec2<f32> };
@vertex fn vs_sky(@builtin(vertex_index) vid : u32) -> SkyVOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : SkyVOut; o.pos = vec4<f32>(p[vid], 0.0, 1.0); o.ndc = p[vid]; return o;
}
@fragment fn fs_sky(in : SkyVOut) -> @location(0) vec4<f32> {
  let dir = normalize(G.camFwd.xyz + G.camRight.xyz * (in.ndc.x * G.camRight.w) + G.camUp.xyz * (in.ndc.y * G.camUp.w));
  let cl = clouds(dir);
  var L = skyRadiance(vec3<f32>(dir.x, max(dir.y, 0.0), dir.z));
  if (dir.y < 0.0) { L = mix(L, G.horizon.rgb * 0.7, smoothstep(0.0, -0.05, dir.y)); }
  if (G.misc.y > 0.5) {
    if (dir.y > 0.0) { L += starField(dir) * G.night.w * smoothstep(0.0, 0.18, dir.y) * (1.0 - cl.a); }
    L += moonHalo(dir);
    L += min(moonDisc(dir), vec3<f32>(400.0)) * (1.0 - cl.a);
  } else {
    L += min(sunDisc(dir), vec3<f32>(400.0)) * (1.0 - cl.a);
  }
  L = mix(L, cl.rgb, cl.a);
  return vec4<f32>(L, 1.0);
}
`;

  // ------------------------------------------------------------------------------------------------ terrain
  CV.wgsl.terrain = () => materials + /* wgsl */`
struct MeshU { info : vec4<f32>, dims : vec4<f32> };
@group(1) @binding(0) var<uniform> M : MeshU;

struct TVOut { @builtin(position) pos : vec4<f32>, @location(0) wpos : vec3<f32> };

// Height of a near-mesh vertex: bare earth plus the tree canopy, which the near mesh carries so ridges have a tree-shaped skyline. A vertex grid
// cannot represent a 5 m dome at a point: the analytic crown height is box-filtered over a footprint of +-r (half the vertex spacing, roughly).
fn nearHeight(xz : vec2<f32>, r : f32, lumps : bool) -> f32 {
  var h = heightAt(xz);
  if (G.tint.w < 0.5) {
    let tf = meanCanopy(xz).y; var ch = 0.0;
    for (var k = 0; k < 4; k++) {
      let q = xz + vec2<f32>(select(-r, r, (k & 1) == 1), select(-r, r, (k & 2) == 2));
      ch += crownH(q, crownAt(q), tf);
    }
    ch *= 0.25;
    if (ch > 0.0 && lumps) {
      // 1 m tiles: the largest leaf clusters (the foliage texture's ~3 m octave, same placement as the shading) bulge the crown's surface, so
      // silhouettes up close are lumpy rather than smooth domes. The texture's exposure channel is high on cluster tops, low in the clefts.
      let cr = crownAt(xz);
      let o = vec2<f32>(cr.id * 41.0, cr.id * 41.0 * 0.61);
      let cs = vec2<f32>(0.9394, 0.3429);
      let qf = vec2<f32>(cs.x * xz.x - cs.y * xz.y, cs.y * xz.x + cs.x * xz.y) / 20.0 + o;
      let top = smoothstep(0.25, 0.75, crownProfile(crownRad(xz, cr), tf));
      ch += (textureSampleLevel(foliageTex, samRep, qf, 2.2).b - 0.5) * 0.8 * top * min(cr.H * 0.25, 1.0);
    }
    if (ch > 0.0) { ch *= treeLine(xz, terrainNormal(xz, 4.0), auxAt(xz), length(G.camPos.xyz - vec3<f32>(xz.x, h, xz.y))); }
    h += max(ch, 0.0);
  }
  return h;
}

@vertex fn vs_terrain(@builtin(vertex_index) vid : u32) -> TVOut {
  let nx = u32(M.dims.x);
  let i = vid % nx; let j = vid / nx;
  let xz = vec2<f32>(M.info.x + (f32(i) + 0.5) * M.info.z, M.info.y + (f32(j) + 0.5) * M.info.w);
  var h : f32;
  if (M.dims.z > 0.5) { h = heightAt(xz) - 0.7 * nearWeight(xz); }   // far mesh: tucked under the near one where they overlap
  else { h = nearHeight(xz, 0.9, false); }
  var o : TVOut;
  o.wpos = vec3<f32>(xz.x, h, xz.y);
  o.pos = G.viewProj * vec4<f32>(o.wpos, 1.0);
  return o;
}

// Near-mesh tiles close to the camera are drawn at 1 m instead of 2 m (one instance per tile), so crowns keep round outlines up close.
// A fine tile's border must meet its coarse neighbours without cracks: border vertices on the 2 m lattice use the coarse height function,
// and the ones between them lie on the straight coarse edge (the mean of their two neighbours).
struct FineTiles { t : array<vec4<f32>, 32> };   // xy: the tile's first vertex (on the 2 m lattice), z: cell (m), w: unused
@group(1) @binding(1) var<uniform> FT : FineTiles;
const FINE_N : u32 = 64u;
@vertex fn vs_terrain_fine(@builtin(vertex_index) vid : u32, @builtin(instance_index) iid : u32) -> TVOut {
  let o = FT.t[iid];
  let kx = vid % (FINE_N + 1u); let kz = vid / (FINE_N + 1u);
  let xz = o.xy + vec2<f32>(f32(kx), f32(kz)) * o.z;
  let edgeZ = kz == 0u || kz == FINE_N; let edgeX = kx == 0u || kx == FINE_N;
  var h : f32;
  if (edgeZ && (kx & 1u) == 1u) { h = 0.5 * (nearHeight(xz - vec2<f32>(o.z, 0.0), 0.9, false) + nearHeight(xz + vec2<f32>(o.z, 0.0), 0.9, false)); }
  else if (edgeX && (kz & 1u) == 1u) { h = 0.5 * (nearHeight(xz - vec2<f32>(0.0, o.z), 0.9, false) + nearHeight(xz + vec2<f32>(0.0, o.z), 0.9, false)); }
  else if (edgeX || edgeZ) { h = nearHeight(xz, 0.9, false); }
  else {
    // the lumps fade in over the first few cells from the border, so the tile still meets its coarse neighbours exactly
    let edge = f32(min(min(kx, FINE_N - kx), min(kz, FINE_N - kz)));
    h = mix(nearHeight(xz, 0.9, false), nearHeight(xz, 0.45, true), smoothstep(1.0, 6.0, edge));
  }
  var out : TVOut;
  out.wpos = vec3<f32>(xz.x, h, xz.y);
  out.pos = G.viewProj * vec4<f32>(out.wpos, 1.0);
  return out;
}

// Radiance of a lit ground point (shared by the terrain pass and by reflections in water).
fn shadeLand(p : vec3<f32>, V : vec3<f32>, dist : f32, withShadow : bool, jit : f32) -> vec3<f32> {
  let sea = seaLevel();
  let pxz = p.xz;
  let ground = heightAt(pxz);                                // bare earth: p.y may sit on a crown
  let eps = max(4.0, dist * 0.004);
  var nG = terrainNormal(pxz, eps);
  if (withShadow && dist < 220.0) {
    // the DEM is bilinear on 4 m cells, so a central difference gives every cell its own flat tilt: up close that reads as faceted planes.
    // Averaging over half-cell offsets smooths the normal across cell edges (faded out with distance, where the facets are sub-pixel; skipped in
    // reflections, which the waves blur anyway).
    let nS = normalize(nG + terrainNormal(pxz + vec2<f32>(2.0, 1.0), eps) + terrainNormal(pxz + vec2<f32>(-1.0, 2.0), eps));
    nG = normalize(mix(nS, nG, smoothstep(140.0, 220.0, dist)));
  }
  let aux = auxAt(pxz);
  let cls = landClass(pxz, nG, aux, dist);
  let mat = landMaterial(pxz, nG, V, dist, cls);
  var n = nG;
  // micro relief: leaf-clump bumps on vegetation, faint ripples on sand; crown domes tilt the normal (mat.nAdd)
  let nb = fadeN(vnoise4(pxz * 1.7), dist, 40.0, 250.0);
  let bump = mix(0.12, 0.03, mat.sandW);
  n = normalize(n + vec3<f32>(nb.r - 0.5, 0.0, nb.g - 0.5) * bump * (1.0 - 0.6 * mat.rockW) + mat.nAdd);
  // wind ripples on dry sand (~14 cm crest spacing, aligned across the wind)
  let ripPh = dot(pxz, vec2<f32>(0.60, -0.80)) * 46.0 + nb.b * 5.0;
  let footS = dist * G.camFwd.w / max(dot(nG, V), 0.06);       // metres of sand per pixel along the view
  let ripK = mat.sandW * (1.0 - smoothstep(0.022, 0.060, footS)) * 0.24;   // 14 cm ripples: gone before they alias into moire
  n = normalize(n + vec3<f32>(0.60, 0.0, -0.80) * cos(ripPh) * ripK);
  // dune-scale undulation (5-20 m) so large sand faces are not perfectly smooth
  let dn = vnoise4(pxz * 0.09);
  n = normalize(n + vec3<f32>(dn.r - 0.5, 0.0, dn.g - 0.5) * 0.10 * mat.sandW * (1.0 - smoothstep(60.0, 500.0, dist)));

  // wetness (swash zone): darkened, glossy, slightly redder sand up to the recent run-up level
  let hw = waveDisplacement(pxz, 0.5, 1.2).y;
  let wetTop = sea + 0.26 + 0.9 * max(hw, 0.0) + 0.5 * abs(hw);
  let wet = (1.0 - smoothstep(wetTop - 0.42, wetTop + 0.08, ground)) * mat.sandW;
  var alb = mat.albedo * mix(vec3<f32>(1.0), vec3<f32>(0.58, 0.54, 0.53), wet);
  // wrack line: seaweed, twigs and shell hash left at the highest recent swash
  let wn = vnoise4(pxz * vec2<f32>(0.55, 0.9));
  let wrack = (1.0 - smoothstep(0.0, 0.10, abs(ground - (sea + 0.46 + 0.10 * (wn.r - 0.5))))) * smoothstep(0.50, 0.78, wn.g) * mat.sandW * (1.0 - smoothstep(60.0, 400.0, dist));
  alb = mix(alb, vec3<f32>(0.075, 0.055, 0.035) * (0.7 + 0.6 * wn.b), wrack * 0.50);

  // leafy edge: where a crown's surface turns away from the viewer the leaf clusters thin out, so the skyline is a fringe, not a smooth arc
  gHole = 0.0;
  if (withShadow && mat.rim > 0.05) {
    let rimK = smoothstep(0.38, 0.03, clamp(dot(n, V), 0.0, 1.0)) * mat.rim;
    let hn = vnoise4(pxz * 3.1 + vec2<f32>(5.1, 9.7));
    if (hn.b * 0.7 + hn.a * 0.3 < rimK * 0.80) {
      // only a real skyline frays: the view ray, continued past this crown, must clear the stand and the hillside beyond it (otherwise something is behind the hole)
      var open = 1.0;
      for (var k = 0; k < 5; k++) {
        let t = 6.0 * pow(3.2, f32(k));                          // 6, 19, 61, 197, 630 m
        let q = p - V * t;
        open = min(open, smoothstep(-0.4, 1.2, q.y - surfaceAt(q.xz)));
      }
      if (open > 0.5) { gHole = 1.0; }
    }
  }

  let L = G.sunDir.xyz;
  let wrap = 0.35 * mat.canopy;                              // leaves transmit: wrapped diffuse on foliage
  let ndl = max((dot(n, L) + wrap) / (1.0 + wrap), 0.0);
  var sh = 1.0;
  if (withShadow && ndl > 0.0) { sh = sunShadow(vec3<f32>(pxz.x, max(p.y, ground), pxz.y) + n * 0.4, jit * 1.2) * structShadow(pxz); }
  if (withShadow && ndl > 0.0) {
    // crowns shade each other and the ground beside them (sand under an overhanging sea grape, litter between trees)
    // (on crowns wherever they are resolved; on the ground only within a few hundred metres, beyond which these shadows are sub-pixel)
    var tfm = 0.0;
    if (mat.leaf > 0.02) { tfm = meanCanopy(pxz).y; }
    else if (dist < 320.0) { tfm = max(meanCanopy(pxz).y, meanCanopy(pxz + normalize(L.xz + vec2<f32>(1e-4, 0.0)) * 6.0).y); }   // here, or trees toward the sun
    if (mat.leaf > 0.02 || tfm > 0.01) {
      sh *= mix(1.0, crownShade(pxz, nG, L, tfm, mat.hTop, jit), max(mat.leaf, 1.0 - mat.canopy));
      // sun flecks: light through gaps in the foliage dapples the shade on the ground
      let fl = vnoise4(pxz * 1.6 + vec2<f32>(2.3, 7.1));
      sh = max(sh, smoothstep(0.64, 0.80, fl.r * 0.7 + fl.g * 0.3) * 0.75 * (1.0 - mat.leaf) * smoothstep(0.02, 0.20, tfm) * (1.0 - smoothstep(60.0, 300.0, dist)));
    }
  }
  let ao = clamp(1.0 - 0.055 * max(aux.y, 0.0) + 0.02 * min(aux.y, 0.0), 0.35, 1.05) * (0.75 + 0.25 * clamp(n.y, 0.0, 1.0)) * mat.occ;
  let Eamb = G.skyE.rgb * (0.5 + 0.5 * n.y) * ao + G.sunE.rgb * (L.y * 0.06 * (0.5 - 0.5 * n.y));
  sh *= pow(clamp(mat.occ, 0.1, 1.5), 0.85);                 // leaf-cluster occlusion also shades the direct light (self-shadowing between clusters)
  var col = alb / PI * (G.sunE.rgb * ndl * sh + Eamb);
  if (mat.leaf > 0.02) {
    // leaves transmit (a crown lit from behind glows yellow-green) and carry a waxy cuticle: a soft sheen of sun on the leaf clusters facing the half vector
    let back = max(dot(-n, L), 0.0);
    col += alb * vec3<f32>(1.0, 1.2, 0.45) * G.sunE.rgb / PI * (0.45 * back * sh * mat.leaf);
    let nh = max(dot(n, normalize(L + V)), 0.0);
    col += G.sunE.rgb * (0.012 * 26.0 / (8.0 * PI) * pow(nh, 18.0) * ndl * sh * mat.leaf);
  }
  // glossy wet film reflecting the sky
  let R = reflect(-V, n);
  let F = fresnelAir(max(dot(n, V), 0.0));
  col += skyWithClouds(vec3<f32>(R.x, abs(R.y), R.z)) * F * wet * 0.85;
  return col;
}

@fragment fn fs_terrain(in : TVOut) -> @location(0) vec4<f32> {
  let p = in.wpos;
  let toCam = G.camPos.xyz - p;
  let dist = length(toCam);
  if (p.y < seaLevel() - 0.6) { return vec4<f32>(0.020, 0.050, 0.060, 1.0); } // submerged: covered by the water pass
  if (G.tint.x > 6.5 && G.tint.x < 7.5) {   // diagnostics: (sand, rock, foliage cover)
    let nG = terrainNormal(p.xz, max(4.0, dist * 0.004)); let cls = landClass(p.xz, nG, auxAt(p.xz), dist);
    let mt = landMaterial(p.xz, nG, toCam / dist, dist, cls);
    return vec4<f32>(mt.sandW, mt.rockW, mt.canopy, 1.0);
  }
  let jit = hash21(in.pos.xy + G.frame.x * vec2<f32>(7.13, 3.71));   // rotates per frame under TAA, which averages it out
  var col = shadeLand(p, toCam / dist, dist, true, jit);
  col = applyFog(col, dist);
  if (gHole > 0.5) { let d = -toCam / dist; col = skyWithClouds(vec3<f32>(d.x, max(d.y, 0.03), d.z)); }
  return vec4<f32>(col, 1.0);
}
`;

  // ------------------------------------------------------------------------------------------------ water
  CV.wgsl.water = (mesh) => materials + CV.wgsl.terrainShade() + /* wgsl */`
const SEG : u32 = ${mesh.seg}u;
const R0 : f32 = ${mesh.r0.toFixed(4)};
const GROWTH : f32 = ${mesh.growth.toFixed(6)};

struct WVOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) rest : vec2<f32>,
  @location(1) wpos : vec3<f32>,
};

@vertex fn vs_water(@builtin(vertex_index) vid : u32) -> WVOut {
  let ring = vid / SEG; let seg = vid % SEG;
  let a = f32(seg) * (TAU / f32(SEG));
  var r = 0.0;
  if (ring > 0u) { r = R0 * pow(GROWTH, f32(ring - 1u)); }
  let rest = G.camPos.xz + r * vec2<f32>(cos(a), sin(a));
  let sea = seaLevel();
  let d = sea - heightAt(rest);
  let spacing = max(r * (GROWTH - 1.0), 0.05);
  var disp = waveDisplacement(rest, d, spacing);
  disp.y += rippleAt(rest).x;
  var o : WVOut;
  o.rest = rest;
  o.wpos = vec3<f32>(rest.x + disp.x, sea + disp.y + 0.012, rest.y + disp.z);
  o.pos = G.viewProj * vec4<f32>(o.wpos, 1.0);
  return o;
}

fn kAbs() -> vec3<f32> { return G.kAbs.rgb; }
// angular softness of reflected edges: wider with distance (unresolved ripple slope scatters the reflection vertically)
fn footprintSoft(dist : f32) -> f32 { return 0.0016 * log2(1.0 + dist * 0.02); }

// Smith masking/shadowing of the reflected ray for a Beckmann/GGX slope distribution with mean-square slope a2. A ray leaving the surface at
// a low elevation is blocked by neighbouring waves with high probability; on a real sea it strikes a second facet and mostly escapes upward
// with far less energy. Only the slope variance carried by the *resolved* geometry is used (the unresolved part already lowers the averaged
// Fresnel term), so distant, fully filtered water is unaffected.
fn smithG1(y : f32, a2 : f32) -> f32 {
  let c2 = max(y * y, 1e-6);
  return 2.0 / (1.0 + sqrt(1.0 + a2 * (1.0 - c2) / c2));
}
fn mssTotalAt(d : f32) -> f32 {
  let e0 = waveEnvC(0, d); let e1 = waveEnvC(1, d);
  return G.shore.x * e0 * e0 + G.shore.y * e1 * e1 + G.shore.z;
}

// Short blue-green flashes of Pyrodinium cells: ~6 cm cells, each flashing at its own random phase for ~0.15 s; the chance of a flash grows with
// the agitation a (0..1). Only meaningful when a pixel is smaller than a cell, so the caller fades it with the pixel footprint.
fn bioSparks(p : vec2<f32>, t : f32, a : f32) -> f32 {
  let q = p / 0.075;
  let i = floor(q);
  let h = hash21(i); let h2 = hash21(i + vec2<f32>(17.3, 5.1));
  let tt = t * (0.7 + 0.8 * h2) + h * 31.0;
  let ph = fract(tt); let cycle = floor(tt);
  let on = step(1.0 - a * 0.42, hash21(i + vec2<f32>(cycle * 1.31, cycle * 0.77)));
  let d = length(fract(q) - 0.5) * 2.0;
  return on * exp(-ph * 9.0) * pow(max(1.0 - d, 0.0), 2.0);
}

// Radiance seen in the reflection of terrain hit at q (cheap: no shadow march).
fn reflectedLand(q : vec3<f32>, R : vec3<f32>, distTotal : f32) -> vec3<f32> {
  let c = shadeLand(q, -R, distTotal, false, 0.0);
  return applyFog(c, distTotal);
}

@fragment fn fs_water(in : WVOut) -> @location(0) vec4<f32> {
  let sea = seaLevel();
  let cam = G.camPos.xyz;
  let t = nowT();
  let P = in.wpos;
  let toCam = cam - P;
  let dist = length(toCam);
  let V = toCam / dist;
  let bedH = heightAt(in.rest);
  let dRest = sea - bedH;

  // ---- wave normal (implicit-LOD sampling happens here, in uniform control flow, before any discard)
  let footAlong = dist * G.camFwd.w / max(V.y, 0.03);          // along-view ground footprint of one pixel (m)
  let ws = waveSlopeAt(in.rest, dRest, footAlong);
  let rp = rippleAt(in.rest);
  var N = normalize(vec3<f32>(-(ws.x + rp.y), 1.0, -(ws.y + rp.z)));
  let ndv0 = dot(N, V);
  if (ndv0 < 0.03) { N = normalize(N + V * (0.03 - ndv0)); }
  let cosv = clamp(dot(N, V), 0.001, 1.0);
  let F = clamp(fresnelAir(cosv) * avgFresnelScale(cosv, ws.w), 0.0, 1.0);
  // Reflected ray. A facet tilted away from the viewer sends the ray below the horizon, where on a real sea it would strike the next
  // wave and bounce again (mostly up into the sky). Mirroring such rays upward approximates that; the old "squash toward the horizon"
  // pinned them exactly at the beach's angular height and drew beach-coloured bars across the shallows.
  var R = reflect(-V, N);
  let downRay = R.y < 0.0;                              // aimed into the next wave, not at the shore
  R = normalize(vec3<f32>(R.x, abs(R.y) + 0.003, R.z));
  // how fast the ray elevation changes between neighbouring pixels (uniform control flow: before the discard): an under-resolved wave field
  // makes adjacent pixels pick unrelated facets, so widen the reflection's coverage ramp by the same amount instead of aliasing
  let dRy = 0.5 * (abs(dpdx(R.y)) + abs(dpdy(R.y)));
  let H = P.y - bedH;                                   // vertical water depth at this point
  let alpha = smoothstep(0.0, 0.05, H);
  if (alpha <= 0.0) { discard; }

  let jit = hash21(in.pos.xy + G.frame.x * vec2<f32>(7.13, 3.71));   // rotates per frame under TAA, which averages it out
  let K = kAbs();

  // ---- reflection: sky, terrain (headlands / forest) and the sun
  // Unresolved ripples (filtered-out slope variance + capillaries finer than the smallest tile) spread each reflection over a lobe a few degrees
  // wide; near the horizon the sky brightens steeply, so a single direction gives hard pale/dark bars. Average the sky over the lobe's spread
  // in elevation (clouds stay a single sample: they are already soft).
  let mssCap = 0.0008 + 0.0006 * G.waveA.z;
  let sigR = min(2.0 * sqrt(0.5 * (ws.w + mssCap)), 0.35);
  let cl = clouds(R);
  let Rlo = normalize(vec3<f32>(R.x, max(R.y - sigR, 0.0), R.z)); let Rhi = normalize(vec3<f32>(R.x, R.y + sigR, R.z));
  let skyAvg = (skyRadiance(Rlo) + 2.0 * skyRadiance(R) + skyRadiance(Rhi)) * 0.25;
  var Lrefl = mix(skyAvg, cl.rgb, cl.a);
  if (G.misc.y > 0.5) { Lrefl = skyWithClouds(R); }               // night: keep the stars' single sample
  if (downRay) {
    // a ray reflected downward strikes the back of the next wave: it sees that water (its body colour) plus a second, weaker reflection of
    // the sky at a typical facet angle. Mirroring it upward instead aimed it at the low beach and striped the shallows with sand-coloured bars.
    let R2 = normalize(vec3<f32>(R.x, R.y + 0.12, R.z));
    Lrefl = mix(G.rDeep.rgb * (G.sunE.rgb * G.sunDir.y + G.skyE.rgb) * 1.6, skyRadiance(R2), fresnelAir(0.2) * 2.2);
  }
  // terrain reflections: rays steeper than ~9 deg can only meet land right beside the shore, so gate the (expensive) march by proximity
  let nearShore = 1.0 - smoothstep(25.0, 110.0, -auxAt(in.rest).x);
  if (!downRay && R.y < mix(0.16, 0.6, nearShore)) {
    // the capillary spread (mssCap, slope variance growing with wind after Cox-Munk) also softens where the land's reflection meets the sky's
    let softR = 0.006 + footprintSoft(dist) + 2.0 * sqrt(0.5 * (ws.w + mssCap)) + 0.8 * dRy;
    let hit = marchLand(P + N * 0.03, R, softR);   // + spread from the slope variance filtered out above
    if (hit.w > 0.0) { Lrefl = mix(Lrefl, reflectedLand(hit.xyz, R, dist + length(hit.xyz - P)), hit.w); }
  }
  // sun glitter: GGX lobe whose width follows the slope variance the mip chain removed
  let Ls = G.sunDir.xyz;
  let Hh = normalize(Ls + V);
  let ndh = max(dot(N, Hh), 0.0); let ndl = max(dot(N, Ls), 0.0);
  let a2 = clamp(0.00012 + 0.5 * ws.w, 0.00012, 0.08);     // GGX width from the slope variance filtering removed
  let dd = ndh * ndh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let vdh = max(dot(V, Hh), 1e-3);
  let Gv = min(1.0, min(2.0 * ndh * cosv / vdh, 2.0 * ndh * ndl / vdh));
  let Fh = fresnelAir(vdh);
  let Lspec = min(G.sunE.rgb * D * Gv * Fh * 0.25 / max(cosv, 0.04), vec3<f32>(3000.0)) * step(0.001, ndl);

  // ---- refraction: march the refracted view ray to the real seabed
  var Lrefr = vec3<f32>(0.0);
  let Tdir = refract(-V, N, 1.0 / N_WATER);
  let TfSun = 1.0 - fresnelAir(G.sunDir.y);
  let Ed = G.sunE.rgb * TfSun * G.sunDir.y + G.skyE.rgb * 0.93;   // total downwelling irradiance just below the surface
  let cosIs = G.sunDir.y;
  let sinTs = sqrt(max(1.0 - cosIs * cosIs, 0.0)) / N_WATER;
  let cosTs = sqrt(max(1.0 - sinTs * sinTs, 0.0));
  let downK = -Tdir.y;
  if (H > 45.0) {
    Lrefr = G.rDeep.rgb * Ed;
  } else {
    let tEnd = clamp((H + 14.0) / max(downK, 0.12), 3.0, 70.0);
    let hitB = marchBed(P, Tdir, tEnd);
    let B = hitB.xyz;
    let depthB = max(sea - B.y, 0.0);
    let nb = terrainNormal(B.xz, 3.0);
    let distB = length(B - cam);
    var alb = seabedAlbedo(B, nb, depthB, distB);
    if (G.tint.z > 0.5) { alb = vec3<f32>(0.68, 0.62, 0.52); }
    // sun through the (flat-approximated) surface, modulated by the caustic map
    let hz = normalize(vec2<f32>(-Ls.x, -Ls.z) + vec2<f32>(1e-5, 0.0));
    let wsun = vec3<f32>(hz.x * sinTs, -cosTs, hz.y * sinTs);
    let footB = distB * G.camFwd.w / max(abs(nb.y), 0.2) * 2.0;
    let penumbra = depthB * 0.0093;                       // sun's 0.53 deg angular diameter softens caustic lines with depth
    let lod = log2(max((footB + penumbra) / max(G.causticB.x, 1e-3), 1.0));
    var C = causticAt(B.xz, lod, distB);
    if (G.tint.z > 0.5) { C = 1.0; }
    let cosBed = max(dot(nb, -wsun), 0.0);
    let Esun = G.sunE.rgb * TfSun * cosIs * (cosBed / cosTs) * exp(-K * (depthB / cosTs)) * C;
    let Esky = G.skyE.rgb * 0.93 * exp(-K * depthB * 1.35) * (0.5 + 0.5 * nb.y);
    let Lbed = alb / PI * (Esun + Esky);
    let pathV = length(B - P);
    let Tup = exp(-K * pathV);
    let kappa = K * (1.0 / cosTs + 1.0 / max(downK, 0.2));
    let Hc = max(P.y - B.y, 0.0);
    let Lcol = G.rDeep.rgb * Ed * (1.0 - exp(-kappa * Hc));
    Lrefr = Lbed * Tup + Lcol;
    if (G.tint.x > 2.5 && G.tint.x < 3.5) { return vec4<f32>(vec3<f32>(C * 0.5), 1.0); }
  }
  Lrefr *= (1.0 - F) / (N_WATER * N_WATER);

  // ---- foam: whitecap Jacobian foam + shoreline lace + disturbed water
  let nf = fbm(in.rest * 2.1 + vec2<f32>(t * 0.045, -t * 0.03), 3);
  let nf2 = vnoise4(in.rest * 6.5 + vec2<f32>(-t * 0.08, t * 0.05)).g;
  let hw = P.y - sea;                                  // instantaneous wave elevation
  let dMean = max(sea - bedH, 0.03);                   // still-water depth
  let band = 1.0 - smoothstep(0.0, 0.12, H);           // thin swash lace hugging the water's edge
  let lace = band * smoothstep(0.34, 0.62, nf + 0.45 * band) * (0.55 + 0.45 * nf2);
  let edgeLine = 1.0 - smoothstep(0.0, 0.035, H);
  // breaking foam on wave crests in very shallow water (breaker index = crest height / depth)
  let brk = smoothstep(0.35, 0.70, hw / dMean) * (1.0 - smoothstep(0.35, 1.0, dMean)) * smoothstep(0.0, 0.04, hw);
  // whitecaps: lacy rather than blobby (noise-modulated), calibrated so coverage follows the observed growth with wind (~1% at 9 m/s, ~4% at 13 m/s)
  let wcap = smoothstep(0.14, 0.72, ws.z * (0.5 + 1.0 * nf2) * 1.3);
  var foam = clamp(wcap + lace * 0.9 + edgeLine * 0.5 + brk * (0.35 + 0.65 * nf) + rp.w * 0.55, 0.0, 1.0);
  foam = foam * foam * (3.0 - 2.0 * foam);
  let Lfoam = vec3<f32>(0.90, 0.92, 0.90) / PI * (G.sunE.rgb * max(dot(N, Ls), 0.0) * 0.9 + G.skyE.rgb * (0.5 + 0.5 * N.y));

  let Gr = smithG1(R.y, max(mssTotalAt(dRest) - ws.w, 0.0));
  var col = Lrefr + F * Gr * Lrefl + Lspec;

  // ---- bioluminescence: Pyrodinium in the top metre of the lagoon flash where the water is disturbed (ripples, wake, paddle) and where waves break
  if (G.bio.x > 0.0) {
    let A = clamp(rp.w + smoothstep(0.08, 0.7, foam) * 0.30, 0.0, 1.0);
    if (A > 0.004) {
      let foot = dist * G.camFwd.w;                                     // metres per pixel: individual sparks only resolve up close
      let sp = bioSparks(in.rest, t, A) * (1.0 - smoothstep(0.05, 0.40, foot));
      let cloud = 0.65 + 0.35 * vnoise4(in.rest * 2.6 + vec2<f32>(t * 0.25, -t * 0.18)).r;          // uneven plankton density
      let thick = 1.0 - exp(-min(H, 1.5) / 0.45);                         // shallow water holds less glowing volume
      let energy = smoothstep(0.02, 0.32, length(rp.yz));                  // brighter where the water is actually moving
      let glow = (0.95 * pow(A, 1.2) * cloud * (0.55 + 0.85 * energy) * thick + 3.4 * sp * thick) * G.bio.x * G.bio.z;
      col += vec3<f32>(0.035, 0.70, 0.95) * glow * (1.0 - F) / (N_WATER * N_WATER) * exp(-K * 0.35);
    }
  }
  // paddle blade marker: a thin pale ring on the water where the blade is
  if (G.paddle.w > 0.5) {
    let pd = length(in.rest - G.paddle.xy);
    let ring = 1.0 - smoothstep(0.0, 0.035, abs(pd - G.paddle.z));
    col += vec3<f32>(0.55, 0.85, 1.0) * ring * (0.55 + 0.45 * smoothstep(0.0, 1.0, G.misc.y)) * mix(0.55, 0.06, G.misc.y);
  }
  col = mix(col, Lfoam, foam * 0.92);
  col = applyFog(col, dist);

  let dbg = G.tint.x;
  if (dbg > 0.5) {
    if (dbg < 1.5) { return vec4<f32>(N * 0.5 + 0.5, 1.0); }
    if (dbg < 2.5) { return vec4<f32>(vec3<f32>(H * 0.1), 1.0); }
    if (dbg > 3.5 && dbg < 4.5) { return vec4<f32>(vec3<f32>(foam), 1.0); }
    if (dbg > 4.5 && dbg < 5.5) { return vec4<f32>(Lrefr * 2.0, 1.0); }
    if (dbg > 5.5) { return vec4<f32>(F * Lrefl, 1.0); }
  }
  return vec4<f32>(col * alpha, alpha);
}
`;

  // The land-shading functions live in the terrain source; expose them for the water module without its vertex/fragment entry points.
  CV.wgsl.terrainShade = () => {
    const src = CV.wgsl.terrain();
    const a = src.indexOf('struct MeshU');
    const b = src.indexOf('// Radiance of a lit ground point');
    const c = src.indexOf('@fragment fn fs_terrain');
    return src.slice(b, c); // shadeLand only (materials already prepended by caller)
  };

  // ------------------------------------------------------------------------------------------------ post (tone mapping + upscale)
  // Two entry paths share one module:  fs = grade + dither straight to the canvas (native resolution);
  //   fs_grade -> LDR texture at the internal resolution, then fs_up = edge-adaptive upscale to the canvas.
  // fs_up estimates the local edge orientation from luma gradients (structure tensor over the 4 centre texels of a 4x4 neighbourhood) and
  // filters with a Lanczos-2-like kernel squeezed across the edge and stretched along it, clamped to the local min/max (no ringing). Stair-steps in
  // a low-resolution frame come out as straight edges, which is what lets the cheap tiers run without MSAA. The idea follows AMD FidelityFX FSR 1
  // (EASU, MIT); the code here is our own compact variant.
  CV.wgsl.post = /* wgsl */`
@group(0) @binding(0) var hdr : texture_2d<f32>;
@group(0) @binding(1) var samP : sampler;
@group(0) @binding(2) var<uniform> PP : array<vec4<f32>, 3>;   // [0] exposure, vignette, time, saturation ; [1] raw-debug flag, source w, source h, night ; [2] sharpen, -
@group(0) @binding(3) var ldr : texture_2d<f32>;
struct VOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vid : u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut; o.pos = vec4<f32>(p[vid], 0.0, 1.0); o.uv = vec2<f32>(p[vid].x * 0.5 + 0.5, 0.5 - p[vid].y * 0.5); return o;
}
fn neutral(color : vec3<f32>) -> vec3<f32> { // Khronos PBR Neutral tone mapper
  let startC = 0.76; let desat = 0.15;
  let x = min(color.r, min(color.g, color.b));
  let off = select(0.04, x - 6.25 * x * x, x < 0.08);
  var c = color - off;
  let peak = max(c.r, max(c.g, c.b));
  if (peak < startC) { return c; }
  let d = 1.0 - startC;
  let newPeak = 1.0 - d * d / (peak + d - startC);
  c *= newPeak / peak;
  let g = 1.0 - 1.0 / (desat * (peak - newPeak) + 1.0);
  return mix(c, vec3<f32>(newPeak), g);
}
fn toSrgb(c : vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92; let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3<f32>(0.0031308));
}
fn grade(raw : vec3<f32>, uv : vec2<f32>) -> vec3<f32> {
  let P = PP[0];
  var c = raw * P.x;
  if (PP[1].w > 0.001) {                                            // night: rod-vision look — desaturate and shift toward blue (Purkinje), highlights keep their colour
    let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let dim = 1.0 - smoothstep(0.25, 2.0, l);
    c = mix(c, vec3<f32>(l) * vec3<f32>(0.78, 0.94, 1.22), 0.50 * PP[1].w * dim);
  }
  let q = uv - 0.5;
  c *= 1.0 - P.y * dot(q, q) * 1.6;
  c = neutral(c);
  let s = toSrgb(c);
  return mix(vec3<f32>(dot(s, vec3<f32>(0.2126, 0.7152, 0.0722))), s, P.w);   // gentle saturation lift (the neutral tone-mapper desaturates)
}
fn dither(pos : vec2<f32>) -> f32 {
  return (fract(sin(dot(pos, vec2<f32>(12.9898, 78.233)) + PP[0].z) * 43758.5453) - 0.5) / 255.0;
}
// After TAA the image is slightly soft (the jitter integrates over the pixel): a light unsharp mask against the four neighbours, clamped to their
// range so it cannot overshoot into halos.
fn fetchSharp(uv : vec2<f32>) -> vec3<f32> {
  let c = textureSampleLevel(hdr, samP, uv, 0.0).rgb;
  let k = PP[2].x;
  if (k <= 0.0) { return c; }
  let px = 1.0 / vec2<f32>(PP[1].y, PP[1].z);
  let a = textureSampleLevel(hdr, samP, uv + vec2<f32>(px.x, 0.0), 0.0).rgb; let b = textureSampleLevel(hdr, samP, uv - vec2<f32>(px.x, 0.0), 0.0).rgb;
  let d = textureSampleLevel(hdr, samP, uv + vec2<f32>(0.0, px.y), 0.0).rgb; let e = textureSampleLevel(hdr, samP, uv - vec2<f32>(0.0, px.y), 0.0).rgb;
  let mn = min(c, min(min(a, b), min(d, e))); let mx = max(c, max(max(a, b), max(d, e)));
  return clamp(c + (c - 0.25 * (a + b + d + e)) * k, mn, mx);
}
@fragment fn fs(in : VOut) -> @location(0) vec4<f32> {   // native resolution: grade + dither to the canvas
  let raw = fetchSharp(in.uv);
  if (PP[1].x > 0.5) { return vec4<f32>(clamp(raw, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0); }   // diagnostics: linear values straight to the framebuffer
  return vec4<f32>(grade(raw, in.uv) + dither(in.pos.xy), 1.0);
}
@fragment fn fs_grade(in : VOut) -> @location(0) vec4<f32> {   // internal resolution: grade into the LDR texture
  let raw = fetchSharp(in.uv);
  if (PP[1].x > 0.5) { return vec4<f32>(clamp(raw, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0); }
  return vec4<f32>(grade(raw, in.uv), 1.0);
}

// windowed-sinc (a = 2) as a polynomial in the squared distance; valid for d2 in [0, 4]
fn kern(d2 : f32) -> f32 {
  let a = 0.4 * d2 - 1.0; let b = 0.25 * d2 - 1.0;
  return 1.5625 * a * a - 0.5625 * b * b;
}
fn lumaOf(c : vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }

@fragment fn fs_up(in : VOut) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(PP[1].y, PP[1].z);
  let hi = vec2<i32>(dims) - vec2<i32>(1);
  let p = in.uv * dims - vec2<f32>(0.5);
  let ip = vec2<i32>(floor(p));
  let fp = p - floor(p);
  var col : array<vec3<f32>, 16>;
  var lum : array<f32, 16>;
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      let c = textureLoad(ldr, clamp(ip + vec2<i32>(i - 1, j - 1), vec2<i32>(0), hi), 0).rgb;
      col[j * 4 + i] = c; lum[j * 4 + i] = lumaOf(c);
    }
  }
  // luma gradients at the four centre texels, blended bilinearly to the sample point -> structure tensor
  let gxf = lum[6] - lum[4];  let gyf = lum[9] - lum[1];
  let gxg = lum[7] - lum[5];  let gyg = lum[10] - lum[2];
  let gxj = lum[10] - lum[8]; let gyj = lum[13] - lum[5];
  let gxk = lum[11] - lum[9]; let gyk = lum[14] - lum[6];
  let wf = (1.0 - fp.x) * (1.0 - fp.y); let wg = fp.x * (1.0 - fp.y); let wj = (1.0 - fp.x) * fp.y; let wk = fp.x * fp.y;
  let jxx = wf * gxf * gxf + wg * gxg * gxg + wj * gxj * gxj + wk * gxk * gxk;
  let jyy = wf * gyf * gyf + wg * gyg * gyg + wj * gyj * gyj + wk * gyk * gyk;
  let jxy = wf * gxf * gyf + wg * gxg * gyg + wj * gxj * gyj + wk * gxk * gyk;
  let tr = jxx + jyy;
  let disc = sqrt(max((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy, 0.0));
  let lam1 = 0.5 * (tr + disc);
  let aniso = disc / max(tr, 1e-6);                        // 0 = isotropic, 1 = a clean straight edge
  let theta = 0.5 * atan2(2.0 * jxy, jxx - jyy);           // direction of the strongest gradient (across the edge)
  let across = vec2<f32>(cos(theta), sin(theta));
  let along = vec2<f32>(-across.y, across.x);
  let edge = smoothstep(0.0004, 0.01, lam1) * aniso;       // only reshape the kernel where a real edge exists
  let sa = 1.0 - 0.45 * edge;                              // < 1: kernel reaches further along the edge
  let sc = 1.0 + 0.40 * edge;                              // > 1: kernel is narrower across it
  var acc = vec3<f32>(0.0); var wsum = 0.0;
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      let v = vec2<f32>(f32(i - 1), f32(j - 1)) - fp;
      let a = dot(v, along) * sa; let c = dot(v, across) * sc;
      let d2 = a * a + c * c;
      if (d2 < 4.0) { let w = kern(d2); acc += col[j * 4 + i] * w; wsum += w; }
    }
  }
  var o = acc / max(wsum, 1e-4);
  let mn = min(min(col[5], col[6]), min(col[9], col[10]));
  let mx = max(max(col[5], col[6]), max(col[9], col[10]));
  o = clamp(o, mn, mx);
  return vec4<f32>(o + dither(in.pos.xy), 1.0);
}
`;
})();
