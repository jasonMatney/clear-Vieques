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

fn sandAlbedo(p : vec2<f32>, dist : f32) -> vec3<f32> {
  let n1 = vnoise4(p * 0.028);
  let grain = fadeN(vnoise4(p * 8.5), dist, 8.0, 70.0);
  let white = vec3<f32>(0.54, 0.51, 0.455);
  let pink = vec3<f32>(0.53, 0.42, 0.37);
  let a = mix(white, pink, smoothstep(0.55, 0.88, n1.g) * 0.55);
  return a * (0.91 + 0.18 * grain.g);
}

struct LandMat { albedo : vec3<f32>, sandW : f32, rockW : f32, vegW : f32, tilt : vec2<f32>, occ : f32 };

fn landMaterial(p : vec3<f32>, n : vec3<f32>, aux : vec4<f32>, dist : f32) -> LandMat {
  let elev = p.y - seaLevel();
  let slope = 1.0 - n.y;
  let sd = aux.x;
  let n1 = vnoise4(p.xz * 0.018);
  let n2 = fadeN(vnoise4(p.xz * 0.11), dist, 900.0, 3500.0);
  let n3 = fadeN(vnoise4(p.xz * 0.75), dist, 60.0, 420.0);
  let n4 = fadeN(vnoise4(p.xz * 3.1), dist, 25.0, 160.0);

  // beach: back-beach limit varies along the shore; the low coastal plain behind it is dry scrub / grass, not sand
  let sandEdge = 9.0 + 26.0 * n1.g + 7.0 * (n3.r - 0.5);          // irregular vegetation line
  var sandW = 1.0 - smoothstep(sandEdge, sandEdge + 1.6, sd);
  sandW *= 1.0 - smoothstep(0.08, 0.24, slope);
  sandW *= 1.0 - smoothstep(3.2, 5.5, elev);
  let rockW = smoothstep(0.17, 0.34, slope) * (0.55 + 0.45 * n2.b) * (1.0 - sandW);

  // dry-forest palette (linear reflectance)
  let leafGreen = vec3<f32>(0.040, 0.056, 0.025);
  let leafDark = vec3<f32>(0.017, 0.029, 0.014);
  let leafDry = vec3<f32>(0.105, 0.088, 0.050);
  let grass = vec3<f32>(0.150, 0.135, 0.070);
  let dryMix = smoothstep(0.35, 0.80, n1.r * 0.55 + n2.g * 0.45);

  // canopy crowns (two octaves), fading to the mean colour with distance
  let cf = crownField(p.xz, 8.0);
  let cf2 = crownField(p.xz + vec2<f32>(31.7, 12.9), 3.4);
  let fadeC = 1.0 - smoothstep(300.0, 1500.0, dist);
  let dome = 1.0 - smoothstep(0.12, 0.66, cf.x);
  let dome2 = 1.0 - smoothstep(0.12, 0.70, cf2.x);
  var crown = mix(leafGreen, leafDry, smoothstep(0.30, 0.95, cf.y) * (0.35 + 0.65 * dryMix));
  crown = mix(crown, leafDark, smoothstep(0.55, 0.0, cf.y) * 0.55);
  let lum = 0.16 + 1.5 * (0.62 * dome + 0.38 * dome2) * (0.65 + 0.7 * n3.r);
  let vegMean = mix(leafGreen, leafDry, dryMix * 0.55) * 0.95;
  var veg = mix(vegMean, crown * lum, fadeC);
  // low, flat coastal plain: grass and scrub
  let plain = (1.0 - smoothstep(1.5, 6.0, elev)) * (1.0 - smoothstep(0.03, 0.10, slope));
  veg = mix(veg, mix(grass, leafDry, 0.35 * n3.g) * (0.75 + 0.5 * n4.r), plain * (0.55 + 0.35 * n2.g));
  // sea-grape / beach scrub band behind the sand: brighter green
  let scrub = smoothstep(sandEdge - 1.0, sandEdge + 3.0, sd) * (1.0 - smoothstep(sandEdge + 6.0, sandEdge + 40.0, sd));
  let cf3 = crownField(p.xz + vec2<f32>(5.3, 71.1), 2.3);
  let clump = (1.0 - smoothstep(0.10, 0.62, cf3.x)) * smoothstep(0.35, 0.65, n2.g + 0.2 * (n3.r - 0.5));
  veg = mix(veg, vec3<f32>(0.058, 0.094, 0.030) * (0.55 + 0.9 * dome2), scrub * 0.55);
  let scrubClump = scrub * clump * (1.0 - fadeC * 0.0);
  let rock = mix(vec3<f32>(0.31, 0.28, 0.23), vec3<f32>(0.19, 0.17, 0.14), n3.g) * (0.70 + 0.50 * n4.r);
  let soil = vec3<f32>(0.30, 0.22, 0.14) * (0.85 + 0.3 * n4.b);
  var alb = mix(veg, rock, rockW);
  alb = mix(alb, sandAlbedo(p.xz, dist), sandW);
  alb = mix(alb, vec3<f32>(0.050, 0.082, 0.026) * (0.5 + 1.0 * (1.0 - cf3.x)), clamp(scrubClump * 1.4, 0.0, 1.0) * (1.0 - smoothstep(400.0, 1200.0, dist) * 0.7));
  let trackW = 1.0 - smoothstep(1.1, 2.3, aux.z);
  alb = mix(alb, soil, trackW * (1.0 - 0.6 * sandW));
  var m : LandMat;
  let vegW = (1.0 - sandW) * (1.0 - rockW) * (1.0 - trackW);
  m.albedo = alb; m.sandW = sandW; m.rockW = rockW; m.vegW = vegW;
  m.tilt = -vec2<f32>(cf.z, cf.w) * 0.55 * dome * fadeC * (1.0 - plain);
  m.occ = mix(1.0, 0.42 + 0.58 * (0.6 * dome + 0.4 * dome2), fadeC * vegW * (1.0 - 0.7 * plain));
  return m;
}

fn seabedAlbedo(q : vec3<f32>, nb : vec3<f32>, depth : f32, dist : f32) -> vec3<f32> {
  let p = q.xz;
  let n1 = vnoise4(p * 0.030);
  let n2 = fadeN(vnoise4(p * 0.19), dist, 200.0, 900.0);
  let n3 = fadeN(vnoise4(p * 1.30), dist, 25.0, 160.0);
  let n4 = fadeN(vnoise4(p * 5.20), dist, 10.0, 70.0);
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
  L += min(sunDisc(dir), vec3<f32>(400.0)) * (1.0 - cl.a);
  L = mix(L, cl.rgb, cl.a);
  return vec4<f32>(L, 1.0);
}
`;

  // ------------------------------------------------------------------------------------------------ terrain
  CV.wgsl.terrain = () => materials + /* wgsl */`
struct MeshU { info : vec4<f32>, dims : vec4<f32> };
@group(1) @binding(0) var<uniform> M : MeshU;

struct TVOut { @builtin(position) pos : vec4<f32>, @location(0) wpos : vec3<f32> };

@vertex fn vs_terrain(@builtin(vertex_index) vid : u32) -> TVOut {
  let nx = u32(M.dims.x);
  let i = vid % nx; let j = vid / nx;
  let xz = vec2<f32>(M.info.x + (f32(i) + 0.5) * M.info.z, M.info.y + (f32(j) + 0.5) * M.info.w);
  var h = heightAt(xz);
  if (M.dims.z > 0.5) { h -= 0.7 * nearWeight(xz); }
  var o : TVOut;
  o.wpos = vec3<f32>(xz.x, h, xz.y);
  o.pos = G.viewProj * vec4<f32>(o.wpos, 1.0);
  return o;
}

// Radiance of a lit ground point (shared by the terrain pass and by reflections in water).
fn shadeLand(p : vec3<f32>, V : vec3<f32>, dist : f32, withShadow : bool, jit : f32) -> vec3<f32> {
  let sea = seaLevel();
  let eps = max(4.0, dist * 0.004);
  var n = terrainNormal(p.xz, eps);
  let aux = auxAt(p.xz);
  let mat = landMaterial(p, n, aux, dist);
  // micro relief: canopy bumps on vegetation, faint ripples on sand
  let nb = fadeN(vnoise4(p.xz * 1.7), dist, 40.0, 250.0);
  let bump = mix(0.16, 0.03, mat.sandW);
  n = normalize(n + vec3<f32>(nb.r - 0.5, 0.0, nb.g - 0.5) * bump * (1.0 - 0.6 * mat.rockW) + vec3<f32>(mat.tilt.x, 0.0, mat.tilt.y));
  // wind ripples on dry sand (~14 cm crest spacing, aligned across the wind)
  let ripPh = dot(p.xz, vec2<f32>(0.60, -0.80)) * 46.0 + nb.b * 5.0;
  let ripK = mat.sandW * (1.0 - smoothstep(30.0, 160.0, dist)) * 0.24;
  n = normalize(n + vec3<f32>(0.60, 0.0, -0.80) * cos(ripPh) * ripK);
  // dune-scale undulation (5-20 m) so large sand faces are not perfectly smooth
  let dn = vnoise4(p.xz * 0.09);
  n = normalize(n + vec3<f32>(dn.r - 0.5, 0.0, dn.g - 0.5) * 0.10 * mat.sandW * (1.0 - smoothstep(60.0, 500.0, dist)));

  // wetness (swash zone): darkened, glossy sand up to the recent run-up level
  let hw = waveDisplacement(p.xz, 0.5, 1.2).y;
  let wetTop = sea + 0.16 + 0.9 * max(hw, 0.0) + 0.5 * abs(hw);
  let wet = (1.0 - smoothstep(wetTop - 0.30, wetTop + 0.10, p.y)) * mat.sandW;
  var alb = mat.albedo * mix(1.0, 0.56, wet);

  let L = G.sunDir.xyz;
  let ndl = max(dot(n, L), 0.0);
  var sh = 1.0;
  if (withShadow && ndl > 0.0) { sh = sunShadow(p + n * 0.4, jit * 1.2); }
  let ao = clamp(1.0 - 0.055 * max(aux.y, 0.0) + 0.02 * min(aux.y, 0.0), 0.35, 1.05) * (0.75 + 0.25 * clamp(n.y, 0.0, 1.0)) * mat.occ;
  let Eamb = G.skyE.rgb * (0.5 + 0.5 * n.y) * ao + G.sunE.rgb * (L.y * 0.06 * (0.5 - 0.5 * n.y));
  sh *= 0.35 + 0.65 * mat.occ;
  var col = alb / PI * (G.sunE.rgb * ndl * sh + Eamb);
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
  let jit = hash21(in.pos.xy);
  var col = shadeLand(p, toCam / dist, dist, true, jit);
  col = applyFog(col, dist);
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
  let H = P.y - bedH;                                   // vertical water depth at this point
  let alpha = smoothstep(0.0, 0.05, H);
  if (alpha <= 0.0) { discard; }

  let jit = hash21(in.pos.xy);
  let K = kAbs();
  var N = normalize(vec3<f32>(-(ws.x + rp.y), 1.0, -(ws.y + rp.z)));
  let ndv0 = dot(N, V);
  if (ndv0 < 0.03) { N = normalize(N + V * (0.03 - ndv0)); }
  let cosv = clamp(dot(N, V), 0.001, 1.0);
  let F = clamp(fresnelAir(cosv) * avgFresnelScale(cosv, ws.w), 0.0, 1.0);

  // ---- reflection: sky, terrain (headlands / forest) and the sun
  var R = reflect(-V, N);
  if (R.y < 0.01) { R = normalize(vec3<f32>(R.x, 0.01 + abs(R.y) * 0.3, R.z)); }
  var Lrefl = skyWithClouds(R);
  // terrain reflections: rays steeper than ~9 deg can only meet land right beside the shore, so gate the (expensive) march by proximity
  let nearShore = 1.0 - smoothstep(25.0, 110.0, -auxAt(in.rest).x);
  if (R.y < mix(0.16, 0.6, nearShore)) {
    let hit = marchLand(P + N * 0.03, R, 0.006 + footprintSoft(dist) + 2.0 * sqrt(0.5 * ws.w));   // + spread from the slope variance filtered out above
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

  var col = Lrefr + F * Lrefl + Lspec;
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

  // ------------------------------------------------------------------------------------------------ post (tone mapping)
  CV.wgsl.post = /* wgsl */`
@group(0) @binding(0) var hdr : texture_2d<f32>;
@group(0) @binding(1) var samP : sampler;
@group(0) @binding(2) var<uniform> P : vec4<f32>;   // exposure, vignette, time, 0
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
@fragment fn fs(in : VOut) -> @location(0) vec4<f32> {
  var c = textureSampleLevel(hdr, samP, in.uv, 0.0).rgb * P.x;
  let q = in.uv - 0.5;
  c *= 1.0 - P.y * dot(q, q) * 1.6;
  c = neutral(c);
  var s = toSrgb(c);
  s = mix(vec3<f32>(dot(s, vec3<f32>(0.2126, 0.7152, 0.0722))), s, P.w);   // gentle saturation lift (the neutral tone-mapper desaturates)
  let n = fract(sin(dot(in.pos.xy, vec2<f32>(12.9898, 78.233)) + P.z) * 43758.5453);
  s += (n - 0.5) / 255.0;
  return vec4<f32>(s, 1.0);
}
`;
})();
