// ClearVieques — shared WGSL prelude (uniforms, bindings, DEM sampling, noise, sky, Fresnel, ray-marchers).
// Frame: x east, y up, z south (metres). Radiometric convention: E_sun(perp) = pi * T_sun, so a white Lambertian
// surface facing the sun has radiance ~T_sun.
(function () {
  'use strict';
  const CV = window.CV;
  CV.wgsl = CV.wgsl || {};

  CV.GLOBALS = [
    ['viewProj', 'mat4'],
    ['camPos', 'vec4'],    // xyz, time (s)
    ['camRight', 'vec4'],  // xyz, tan(fovX/2)
    ['camUp', 'vec4'],     // xyz, tan(fovY/2)
    ['camFwd', 'vec4'],    // xyz, pixel angle (rad)
    ['screen', 'vec4'],    // w, h, 1/w, 1/h
    ['sunDir', 'vec4'],    // xyz -> sun, elevation (rad)
    ['sunE', 'vec4'],      // rgb direct normal irradiance (pi * transmittance), exposure
    ['skyE', 'vec4'],      // rgb sky irradiance on a horizontal plane, unused
    ['horizon', 'vec4'],   // rgb horizon radiance, air extinction (1/m)
    ['kAbs', 'vec4'],      // rgb attenuation K (1/m), sea level (m)
    ['rDeep', 'vec4'],     // rgb deep-water reflectance, turbidity 0..1
    ['waveA', 'vec4'],     // choppy, foam gain, wind, energy
    ['casc0', 'vec4'],     // L, cos(rot), sin(rot), weight
    ['casc1', 'vec4'],
    ['casc2', 'vec4'],
    ['demNear', 'vec4'],   // x0, z0, sizeX, sizeZ
    ['demFar', 'vec4'],
    ['demInfo', 'vec4'],   // blend width, unused...
    ['ripple', 'vec4'],    // centre x, centre z, window size (m), enabled
    ['caustic', 'vec4'],   // origin x, origin z, size (m), intensity
    ['causticB', 'vec4'],  // texel (m), max lod, fade start (m), fade end (m)
    ['shore', 'vec4'],     // swash amplitude, ...
    ['misc', 'vec4'],      // quality, night, moon, site material set (0 = dry-forest coast, 1 = mangrove lagoon)
    ['tint', 'vec4'],      // debug view id, cloud cover, optics test, unused
    ['crownA', 'vec4'],    // crown map: x0, z0, texel (m), nx
    ['crownB', 'vec4'],    // nz, mean-canopy cell (m), max canopy height (m), unused
    ['night', 'vec4'],     // moon relative illuminance, signed phase angle (rad, + waxing), light gain K, star visibility
    ['bio', 'vec4'],       // bioluminescence: site intensity, spark rate, glow gain, -
    ['paddle', 'vec4'],    // virtual paddle blade: x, z, radius (m), active
    ['frame', 'vec4'],     // TAA: frame index (mod 64, seeds per-pixel noise), jitter x, y (NDC), enabled
    ['windV', 'vec4'],     // direction the wind blows toward (x, z, unit), speed (m/s), -
    ['trees', 'vec4'],     // 3D trees near the camera: fade start, fade end (m from the camera to the crown centre), enabled, -
    ['st0', 'vec4'], ['st1', 'vec4'], ['st2', 'vec4'], ['st3', 'vec4'],   // OSM shelters: x, z, yaw, roof half extent (0 = unused)
    ['st4', 'vec4'], ['st5', 'vec4'], ['st6', 'vec4'], ['st7', 'vec4'],
  ];
  CV.globals = new CV.UniformBlock(CV.GLOBALS);

  CV.wgsl.prelude = function () {
    return /* wgsl */`
${CV.globals.wgsl('Globals')}

@group(0) @binding(0)  var<uniform> G : Globals;
@group(0) @binding(1)  var samLin : sampler;
@group(0) @binding(2)  var samRep : sampler;
@group(0) @binding(3)  var demNearTex : texture_2d<f32>;
@group(0) @binding(4)  var demFarTex : texture_2d<f32>;
@group(0) @binding(5)  var auxTex : texture_2d<f32>;
@group(0) @binding(6)  var skyLut : texture_2d<f32>;
@group(0) @binding(7)  var waveDisp : texture_2d_array<f32>;
@group(0) @binding(8)  var waveSlope : texture_2d_array<f32>;
@group(0) @binding(9)  var rippleTex : texture_2d<f32>;
@group(0) @binding(10) var causticTex : texture_2d<f32>;
@group(0) @binding(11) var noiseTex : texture_2d<f32>;
@group(0) @binding(12) var crownTex : texture_2d<f32>;   // tree crowns, 2 m texels (rgba8): centre offset x, z, crown height, id
@group(0) @binding(13) var meanTex : texture_2d<f32>;    // stand mean, 4 m texels on the near-DEM grid (rgba8): mean canopy height, tree fraction, -, -
@group(0) @binding(14) var foliageTex : texture_2d<f32>; // leaf-cluster structure, tileable + mip-mapped (rgba8): slope x, slope z, occlusion, tint

const PI : f32 = 3.14159265359;
const TAU : f32 = 6.28318530718;
const N_WATER : f32 = 1.333;
const SUN_RADIUS : f32 = 0.00465;

fn sat(x : f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sat3(x : vec3<f32>) -> vec3<f32> { return clamp(x, vec3<f32>(0.0), vec3<f32>(1.0)); }
fn luma(c : vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn seaLevel() -> f32 { return G.kAbs.w; }
fn nowT() -> f32 { return G.camPos.w; }

// ------------------------------------------------------------------ DEM (near 4 m grid blended into far 20 m grid)
fn nearWeight(p : vec2<f32>) -> f32 {
  let o = G.demNear.xy; let s = G.demNear.zw;
  let d = min(min(p.x - o.x, o.x + s.x - p.x), min(p.y - o.y, o.y + s.y - p.y));
  return smoothstep(0.0, G.demInfo.x, d);
}
// The far grid (~14 x 10 km) is all the elevation data there is. Past its edge is treated as open sea, and the land sinks gently into it over the
// last ~1-2 km (along an irregular line), instead of the clamped edge values running to the horizon as straight strips of land and sea.
fn farEdgeFade(p : vec2<f32>) -> f32 {
  let o = G.demFar.xy; let s = G.demFar.zw;
  let d = min(min(p.x - o.x, o.x + s.x - p.x), min(p.y - o.y, o.y + s.y - p.y));
  return smoothstep(150.0, 1700.0, d + 900.0 * (vnoise4(p * 0.0006 + vec2<f32>(2.7, 8.1)).r - 0.5));
}
fn heightFar(p : vec2<f32>) -> f32 {
  let h = textureSampleLevel(demFarTex, samLin, (p - G.demFar.xy) / G.demFar.zw, 0.0).r;
  return mix(-25.0, h, farEdgeFade(p));
}
fn heightNear(p : vec2<f32>) -> f32 {
  return textureSampleLevel(demNearTex, samLin, (p - G.demNear.xy) / G.demNear.zw, 0.0).r;
}
fn heightAt(p : vec2<f32>) -> f32 {
  let w = nearWeight(p);
  if (w >= 1.0) { return heightNear(p); }                     // inside the near window the far grid is fully blended out: skip its fetch
  var h = heightFar(p);
  if (w > 0.0) { h = mix(h, heightNear(p), w); }
  return h;
}
fn terrainNormal(p : vec2<f32>, e : f32) -> vec3<f32> {
  let hl = heightAt(p - vec2<f32>(e, 0.0)); let hr = heightAt(p + vec2<f32>(e, 0.0));
  let hd = heightAt(p - vec2<f32>(0.0, e)); let hu = heightAt(p + vec2<f32>(0.0, e));
  return normalize(vec3<f32>(hl - hr, 2.0 * e, hd - hu));
}
// (shoreDist [m, + inland], curvature, trackDist [m], _) — only meaningful inside the near window
fn auxAt(p : vec2<f32>) -> vec4<f32> {
  let uv = (p - G.demNear.xy) / G.demNear.zw;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { return vec4<f32>(max(heightFar(p), 0.0) * 12.0 - 30.0, 0.0, 64.0, 0.0); }
  return textureSampleLevel(auxTex, samLin, uv, 0.0);
}

// ------------------------------------------------------------------ tree canopy (baked by CV.Canopy from the DEM + land classification)
// Each 2 m texel of crownTex holds the nearest crown: offset to its centre (metres, +-CROWN_OFF/2), crown height (/CANOPY_MAX) and a random id.
// A crown is an analytic dome: radius from the id, h(r) = H * (0.5 * floor(r) + 0.5 * (1 - r^2)^0.75); the floor (foliage between crowns) fades out
// beyond ~1.9 R so open ground next to a crown stays at ground level.
const CROWN_OFF : f32 = 32.0;   // a texel may belong to a crown whose centre lies in a neighbouring lattice cell (the tallest one there), up to +-16 m away
// Crowns are slightly elliptical and each has its own orientation (rot = cos, sin; iR = 1 / radius along the crown's two axes), so the canopy is
// not a lattice of identical round bumps. The shape follows from the id alone, so the bake and the shaders agree on it.
struct Crown { c : vec2<f32>, H : f32, R : f32, id : f32, rot : vec2<f32>, iR : vec2<f32> };
fn crownShape(c : vec2<f32>, H : f32, id : f32) -> Crown {
  var cr : Crown; cr.c = c; cr.H = H; cr.id = id;
  cr.R = 3.0 + 2.4 * fract(id * 7.31 + 0.17);
  let sq = 0.80 + 0.20 * fract(id * 23.7 + 0.41);                               // squash: 0.8 .. 1.0 along the minor axis
  cr.rot = normalize(vec2<f32>(fract(id * 37.13 + 0.21), fract(id * 61.71 + 0.57)) * 2.0 - 1.0 + vec2<f32>(0.001, 0.0));   // orientation without trig
  cr.iR = vec2<f32>(1.0 / (cr.R * (1.08 + 0.10 * fract(id * 5.3))), 1.0 / (cr.R * sq * 0.92));
  return cr;
}
fn crownAt(p : vec2<f32>) -> Crown {
  let t = (p - G.crownA.xy) / G.crownA.z;
  if (t.x < 0.0 || t.y < 0.0 || t.x >= G.crownA.w || t.y >= G.crownB.x) { return crownShape(p, 0.0, 0.0); }
  let ti = vec2<i32>(floor(t));
  let d = textureLoad(crownTex, ti, 0);
  return crownShape((vec2<f32>(ti) + 0.5) * G.crownA.z + G.crownA.xy + (d.xy - 0.5) * CROWN_OFF, d.z * G.crownB.z, d.w);
}
// 1 where a crown is drawn as a 3D tree (CV.Trees), 0 where it is a dome on the terrain; in between both, dissolving into each other
fn tree3DW(c : vec2<f32>) -> f32 {
  if (G.trees.z < 0.5) { return 0.0; }
  return 1.0 - smoothstep(G.trees.x, G.trees.y, length(c - G.camPos.xz));
}
// distance from the crown centre in crown radii (elliptical metric)
fn crownRad(p : vec2<f32>, cr : Crown) -> f32 {
  let d = p - cr.c;
  return length(vec2<f32>(d.x * cr.rot.x + d.y * cr.rot.y, -d.x * cr.rot.y + d.y * cr.rot.x) * cr.iR);
}
// tf = local tree fraction (foliage between crowns exists only inside a dense stand, never beside open sand)
fn crownProfile(r : f32, tf : f32) -> f32 {
  let dome = pow(max(1.0 - pow(r, 2.4), 0.0), 0.55);         // flat-topped, steep-sided (umbrella crowns): survives 2 m mesh sampling without turning into cones
  return 0.50 * (1.0 - smoothstep(1.30, 2.1, r)) * smoothstep(0.55, 0.90, tf) + 0.50 * dome;
}
fn crownH(p : vec2<f32>, cr : Crown, tf : f32) -> f32 {
  if (cr.H <= 0.0) { return 0.0; }
  return cr.H * crownProfile(crownRad(p, cr), tf);
}
// surface slope (dh/dx, dh/dz) of the crown dome at p (finite difference of the analytic profile; capped near the rim)
fn crownGrad(p : vec2<f32>, cr : Crown, tf : f32) -> vec2<f32> {
  if (cr.H <= 0.0) { return vec2<f32>(0.0); }
  let e = 0.35;
  let hx = crownH(p + vec2<f32>(e, 0.0), cr, tf) - crownH(p - vec2<f32>(e, 0.0), cr, tf);
  let hz = crownH(p + vec2<f32>(0.0, e), cr, tf) - crownH(p - vec2<f32>(0.0, e), cr, tf);
  return clamp(vec2<f32>(hx, hz) / (2.0 * e), vec2<f32>(-2.5), vec2<f32>(2.5));
}
// smooth stand height (m) and tree fraction on the 4 m near-DEM grid: used for shadows, reflections and the far view
// Outside the near window there are no baked crowns: Vieques' hills are mostly dry forest, so the stand there is estimated from the far DEM
// (forest above the coastal flats, thinner on the highest ground) with stand-scale variation. Blended into the baked stand at the window edge.
fn farCanopy(p : vec2<f32>) -> vec2<f32> {
  let e = heightFar(p) - seaLevel();
  let n = vnoise4(p * 0.0035 + vec2<f32>(5.5, 1.2));
  let tf = smoothstep(1.5, 7.0, e) * (1.0 - 0.35 * smoothstep(90.0, 220.0, e)) * smoothstep(0.25, 0.55, n.r * 0.8 + 0.2 * n.g + 0.12);
  return vec2<f32>(tf * (4.0 + 2.0 * n.b), tf);
}
fn meanCanopy(p : vec2<f32>) -> vec2<f32> {
  let uv = (p - G.demNear.xy) / G.demNear.zw;
  let w = nearWeight(p);
  var near = vec2<f32>(0.0);
  if (uv.x >= 0.0 && uv.y >= 0.0 && uv.x <= 1.0 && uv.y <= 1.0) {
    let m = textureSampleLevel(meanTex, samLin, uv, 0.0);
    near = vec2<f32>(m.x * G.crownB.z, m.y);
    if (w >= 1.0) { return near; }
  }
  return mix(farCanopy(p), near, w);
}
// Shore character (baked with the canopy mean, 4 m): exposure to the waves 0..1, rockiness 0..1. Away from the near window: open, not rocky.
fn shoreInfo(p : vec2<f32>) -> vec2<f32> {
  let uv = (p - G.demNear.xy) / G.demNear.zw;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { return vec2<f32>(1.0, 0.0); }
  return textureSampleLevel(meanTex, samLin, uv, 0.0).ba;
}
// Bare-earth DEM plus the smoothed canopy: what the sun and reflected rays actually meet.
fn surfaceAt(p : vec2<f32>) -> f32 { return heightAt(p) + meanCanopy(p).x; }

// Soft roof shadow of an OSM picnic shelter on the ground (1 = sunlit, 0.3 = shaded): the roof rectangle projected along the sun.
fn structOne(s : vec4<f32>, p : vec2<f32>) -> f32 {
  if (s.w <= 0.0) { return 1.0; }
  let L = G.sunDir.xyz;
  let off = -L.xz / max(L.y, 0.05) * 3.2;                     // roof plane ~3.2 m up
  let d = p - (s.xy + off);
  let c = cos(s.z); let sn = sin(s.z);
  let q = vec2<f32>(d.x * c + d.y * sn, -d.x * sn + d.y * c);
  let e = max(abs(q.x), abs(q.y)) - s.w;
  return mix(0.30, 1.0, smoothstep(-0.10, 0.45, e));
}
fn structShadow(p : vec2<f32>) -> f32 {
  return min(min(min(structOne(G.st0, p), structOne(G.st1, p)), min(structOne(G.st2, p), structOne(G.st3, p))),
             min(min(structOne(G.st4, p), structOne(G.st5, p)), min(structOne(G.st6, p), structOne(G.st7, p))));
}

// ------------------------------------------------------------------ noise (smooth value noise from a random RGBA8 texture)
fn vnoise4(p : vec2<f32>) -> vec4<f32> {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return textureSampleLevel(noiseTex, samRep, (i + 0.5 + u) / 256.0, 0.0);
}
fn fbm(p : vec2<f32>, oct : i32) -> f32 {
  var a = 0.5; var s = 0.0; var q = p; var norm = 0.0;
  for (var i = 0; i < oct; i++) { s += a * vnoise4(q).r; norm += a; q = q * 2.03 + vec2<f32>(17.1, 9.2); a *= 0.5; }
  return s / norm;
}
fn hash21(p : vec2<f32>) -> f32 {
  var q = fract(p * vec2<f32>(0.1031, 0.1030));
  q += dot(q, q.yx + 33.33);
  return fract((q.x + q.y) * q.x);
}

// ------------------------------------------------------------------ sky
fn skyLutUV(dir : vec3<f32>) -> vec2<f32> {
  let sh = normalize(vec2<f32>(G.sunDir.x, G.sunDir.z) + vec2<f32>(1e-5, 0.0));
  let vh = normalize(vec2<f32>(dir.x, dir.z) + vec2<f32>(1e-5, 0.0));
  let phi = acos(clamp(dot(sh, vh), -1.0, 1.0));
  let e = asin(clamp(dir.y, 0.0, 1.0));
  return vec2<f32>(phi / PI, sqrt(e / (0.5 * PI)));
}
fn skyRadiance(dir : vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(skyLut, samLin, skyLutUV(dir), 0.0).rgb * G.skyE.w;   // skyE.w = light gain (1 by day, the moon-phase gain at night)
}
fn sunDisc(dir : vec3<f32>) -> vec3<f32> {
  let c = dot(dir, G.sunDir.xyz);
  let a2 = 2.0 * (1.0 - c);
  let r2 = SUN_RADIUS * SUN_RADIUS;
  let edge = 1.0 - smoothstep(r2 * 0.55, r2 * 1.15, a2);
  let mu = sqrt(max(1.0 - a2 / r2, 0.0));
  return G.sunE.rgb / (PI * r2) * edge * (0.62 + 0.38 * mu);
}

// ------------------------------------------------------------------ night sky: stars, Milky Way, moon
fn hash31(p : vec3<f32>) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.zyx + 31.32);
  return fract((q.x + q.y) * q.z);
}
// One layer of a cell-hash star field on the unit sphere (one star per cell, kept inside the cell so no neighbour search is needed).
fn starLayer(dir : vec3<f32>, scale : f32, seed : f32, density : f32) -> f32 {
  let p = dir * scale;
  let ip = floor(p); let fp = p - ip;
  let h = hash31(ip + seed);
  if (h < density) { return 0.0; }
  let o = 0.22 + 0.56 * vec3<f32>(hash31(ip + 11.1 + seed), hash31(ip + 23.7 + seed), hash31(ip + 47.3 + seed));
  let ang = length(fp - o) / scale;                                    // angular distance to the star (rad)
  let r = max(G.camFwd.w * 1.25, 0.0004);                              // ~1.25 pixels: stars stay point-like at any field of view
  let mag = pow((h - density) / (1.0 - density), 5.0);                 // few bright stars, many faint ones
  return mag * (1.0 - smoothstep(0.0, r, ang)) * (0.55 + 0.9 * hash31(ip + 5.5 + seed));
}
fn starField(dir : vec3<f32>) -> vec3<f32> {
  var s = 0.9 * starLayer(dir, 55.0, 1.0, 0.86) + 0.55 * starLayer(dir, 120.0, 7.0, 0.90) + 0.32 * starLayer(dir, 260.0, 13.0, 0.93);
  // twinkle is skipped (the sky is static); colour: mostly blue-white, a few warm
  let warm = hash31(floor(dir * 55.0) + 3.0);
  var col = mix(vec3<f32>(0.75, 0.86, 1.0), vec3<f32>(1.0, 0.86, 0.68), smoothstep(0.7, 1.0, warm));
  // Milky Way: a soft band along a tilted great circle with dark lanes
  let gn = normalize(vec3<f32>(0.35, 0.82, 0.45));
  let band = exp(-pow(dot(dir, gn) / 0.20, 2.0));
  let lane = vnoise4(dir.xz * 5.0 + dir.y * vec2<f32>(3.0, 7.0)).r;          // continuous on the sphere (an atan2 azimuth would leave a seam)
  let mw = band * (0.25 + 1.1 * smoothstep(0.30, 0.75, lane)) * 0.05;
  return col * s * 1.6 + vec3<f32>(0.62, 0.72, 1.0) * mw;
}
// Moon disc with the true phase: a sphere lit from a direction that rotates with the phase angle, maria as dark patches, a hint of earthshine.
const MOON_R : f32 = 0.0068;   // angular radius (rad): the real 0.0045 enlarged 1.5x, as the moon appears to the eye
fn moonDisc(dir : vec3<f32>) -> vec3<f32> {
  let M = G.sunDir.xyz;
  if (dot(dir, M) < 0.9994) { return vec3<f32>(0.0); }
  let r = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), M)); let u = cross(M, r);
  let x = dot(dir, r) / MOON_R; let y = dot(dir, u) / MOON_R;
  let d2 = x * x + y * y;
  if (d2 > 1.0) { return vec3<f32>(0.0); }
  let z = sqrt(1.0 - d2);
  let a = G.night.y; let al = abs(a); let sg = select(-1.0, 1.0, a >= 0.0);
  let nl = x * sg * sin(al) + z * cos(al);                             // n . l on the lunar sphere
  let lit = smoothstep(-0.03, 0.10, nl);
  let mar = 0.60 + 0.40 * smoothstep(0.30, 0.68, vnoise4(vec2<f32>(x, y) * 2.4 + 4.0).r);
  let edge = 1.0 - smoothstep(0.96, 1.0, sqrt(d2));                    // soft limb
  let base = (lit * (0.80 + 0.20 * z) + 0.0007) * mar * edge;               // 0.0007: earthshine on the dark limb
  return G.sunE.rgb / (PI * MOON_R * MOON_R) * 0.12 * base;
}
// scattered glow around the moon (lens and atmosphere): larger and brighter for a fuller moon
fn moonHalo(dir : vec3<f32>) -> vec3<f32> {
  let th = acos(clamp(dot(dir, G.sunDir.xyz), -1.0, 1.0));
  return G.sunE.rgb * (0.030 * exp(-th * 22.0) + 0.0016 / (1.0 + pow(th / 0.16, 2.0)));
}

// ------------------------------------------------------------------ fair-weather cumulus (planar layer, procedural)
fn cloudDensity(p : vec2<f32>, detail : bool) -> f32 {
  let q = p * (1.0 / 5200.0);
  let base = fbm(q + vec2<f32>(0.37, 0.11), 3);
  var det = 0.5;
  if (detail) { det = fbm(q * 6.1 + vec2<f32>(7.1, 2.3), 2); }
  let d = base - (1.0 - G.tint.y) * 0.95 + (det - 0.5) * 0.55;
  let x = clamp(d * 4.2, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}
// rgb radiance + alpha. Thick parts facing away from the sun are darker; thin toward the horizon.
fn clouds(dir : vec3<f32>) -> vec4<f32> {
  if (dir.y < 0.02 || G.tint.y < 0.01) { return vec4<f32>(0.0); }
  let t = (2800.0 - G.camPos.y) / dir.y;
  let p = G.camPos.xz + dir.xz * t + vec2<f32>(nowT() * -7.0, nowT() * 2.0);
  let d = cloudDensity(p, true);
  if (d <= 0.002) { return vec4<f32>(0.0); }
  let sunH = normalize(vec2<f32>(G.sunDir.x, G.sunDir.z) + vec2<f32>(1e-5, 0.0));
  let d2 = cloudDensity(p + sunH * 300.0, false);
  let lit = clamp(1.0 - (d2 - d) * 1.7 - 0.18 * d, 0.0, 1.0);
  let amb = G.skyE.rgb * 0.95 / PI;
  let sun = G.sunE.rgb * max(G.sunDir.y, 0.25) * 0.90 / PI;
  let col = amb * (0.55 + 0.45 * (1.0 - d)) + sun * lit * (0.50 + 0.50 * (1.0 - 0.5 * d));
  let fade = smoothstep(0.05, 0.30, dir.y);
  return vec4<f32>(mix(G.horizon.rgb * 1.05, col, fade * 0.9 + 0.1), d * fade);
}
// Fraction of direct sun (or moon) light reaching p through the cloud layer: the sun ray from p crosses the 2800 m layer where the sky pass
// draws the cloud (same drift), so shadows move with the clouds overhead. Cumulus let little direct light through their cores.
fn cloudShadow(p : vec3<f32>) -> f32 {
  if (G.tint.y < 0.01 || G.sunDir.y < 0.03) { return 1.0; }
  let L = G.sunDir.xyz;
  let q = p.xz + L.xz * ((2800.0 - p.y) / L.y) + vec2<f32>(nowT() * -7.0, nowT() * 2.0);
  return 1.0 - 0.85 * cloudDensity(q, true);
}
fn skyWithClouds(dir : vec3<f32>) -> vec3<f32> {
  let c = clouds(dir);
  var L = skyRadiance(dir);
  if (G.misc.y > 0.5 && dir.y > 0.0) {                                  // night: stars fade toward the horizon haze, and behind clouds
    L += starField(dir) * G.night.w * smoothstep(0.0, 0.18, dir.y) * (1.0 - c.a);
  }
  return mix(L, c.rgb, c.a);
}

// ------------------------------------------------------------------ optics helpers
fn fresnelAir(cosi : f32) -> f32 { // unpolarised, air -> water
  let c = clamp(cosi, 0.0, 1.0);
  let sin2t = (1.0 - c * c) / (N_WATER * N_WATER);
  let cost = sqrt(max(1.0 - sin2t, 0.0));
  let rs = (c - N_WATER * cost) / (c + N_WATER * cost + 1e-6);
  let rp = (N_WATER * c - cost) / (N_WATER * c + cost + 1e-6);
  return 0.5 * (rs * rs + rp * rp);
}

// Fresnel reflectance averaged over a Gaussian slope distribution with total mean-square slope s (Bruneton, Neyret & Poulin 2010 fit),
// expressed as a scale on the exact smooth-surface value so it reduces to 1 when s = 0.
fn avgFresnelScale(c : f32, s : f32) -> f32 {
  let f0 = 0.02;
  let cc = clamp(c, 0.0, 1.0);
  let smooth1 = f0 + (1.0 - f0) * pow(1.0 - cc, 5.0);
  let avg = f0 + (1.0 - f0) * pow(1.0 - cc, 5.0 * exp(-2.69 * s)) / (1.0 + 22.7 * pow(s, 1.5));
  return avg / max(smooth1, 1e-4);
}

// ------------------------------------------------------------------ wave cascades (3 tiled FFT patches, each rotated/offset to hide tiling)
fn casc(c : i32) -> vec4<f32> {
  if (c == 0) { return G.casc0; }
  if (c == 1) { return G.casc1; }
  return G.casc2;
}
fn cascUV(cc : vec4<f32>, xz : vec2<f32>, k : f32) -> vec2<f32> {
  let q = vec2<f32>(cc.y * xz.x + cc.z * xz.y, -cc.z * xz.x + cc.y * xz.y);
  return q / cc.x + vec2<f32>(0.313, 0.577) * k;
}
fn cascRotBack(cc : vec4<f32>, v : vec2<f32>) -> vec2<f32> {
  return vec2<f32>(cc.y * v.x - cc.z * v.y, cc.z * v.x + cc.y * v.y);
}
// Gravity waves die away in the shallows; ripples persist. d = water depth (m).
fn waveEnvC(c : i32, d : f32) -> f32 {
  if (c == 0) { return 0.25 + 0.75 * smoothstep(0.0, 1.4, d); }
  if (c == 1) { return 0.45 + 0.55 * smoothstep(0.0, 0.8, d); }
  return 1.0;
}
// Vertex/photon-side: (dx, height, dz) at rest position xz with explicit LOD (from vertex spacing).
fn waveDisplacement(xz : vec2<f32>, d : f32, spacing : f32) -> vec3<f32> {
  var r = vec3<f32>(0.0);
  for (var c = 0; c < 3; c++) {
    let cc = casc(c);
    let texel = cc.x / 256.0;
    let lod = clamp(log2(max(spacing, 1e-4) / texel), 0.0, 8.0);
    let s = textureSampleLevel(waveDisp, samRep, cascUV(cc, xz, f32(c)), c, lod);
    let e = waveEnvC(c, d) * cc.w;
    let dxz = cascRotBack(cc, vec2<f32>(s.x, s.z));
    r += e * vec3<f32>(dxz.x, s.y, dxz.y);
  }
  return r;
}
// Slopes (dh/dx, dh/dz) with explicit LOD; used by photons (compute) — returns xy = slope, z = height.
fn waveSlopeLevel(xz : vec2<f32>, d : f32, spacing : f32) -> vec3<f32> {
  var s2 = vec2<f32>(0.0); var h = 0.0;
  for (var c = 0; c < 3; c++) {
    let cc = casc(c);
    let texel = cc.x / 256.0;
    let lod = clamp(log2(max(spacing, 1e-4) / texel), 0.0, 8.0);
    let uv = cascUV(cc, xz, f32(c));
    let t = textureSampleLevel(waveSlope, samRep, uv, c, lod);
    let hh = textureSampleLevel(waveDisp, samRep, uv, c, lod).y;
    let e = waveEnvC(c, d) * cc.w;
    s2 += cascRotBack(cc, t.xy) * e; h += hh * e;
  }
  return vec3<f32>(s2, h);
}
// Fragment-side (implicit LOD + anisotropic filtering): xy = slope, z = foam, w = slope variance removed by filtering (both axes).
// footAlong = along-view ground footprint of a pixel (m). Cascades whose texels are much smaller than that are faded out (their
// variance is returned in .w so the caller can widen roughness / average the Fresnel term instead of aliasing).
fn waveSlopeAt(xz : vec2<f32>, d : f32, footAlong : f32) -> vec4<f32> {
  var s2 = vec2<f32>(0.0); var foam = 0.0; var lost = 0.0;
  for (var c = 0; c < 3; c++) {
    let cc = casc(c);
    let texel = cc.x / 256.0;
    let keep = 1.0 - smoothstep(4.0 * texel, 16.0 * texel, footAlong);
    let t = textureSample(waveSlope, samRep, cascUV(cc, xz, f32(c)), c);
    let e = waveEnvC(c, d) * cc.w;
    s2 += cascRotBack(cc, t.xy) * e * keep;
    foam = max(foam, t.w * e);
    let mss = select(select(G.shore.z, G.shore.y, c == 1), G.shore.x, c == 0);
    lost += (1.0 - keep * keep) * mss * e * e;
  }
  return vec4<f32>(s2, foam, lost);
}

// ------------------------------------------------------------------ wind in the vegetation
// Horizontal sway (m per m of height above the ground) at p: a steady lean downwind plus gusts that travel downwind across the land as patches
// a few tens of metres wide, each plant's own sway frequency, and a little turbulence. Grows with the wind speed.
fn windGust(p : vec2<f32>, t : f32) -> f32 {                        // 0..1, a pattern of gusts advected downwind
  let gq = p - G.windV.xy * (t * G.windV.z * 0.7);
  return smoothstep(0.30, 0.85, vnoise4(gq * 0.018 + vec2<f32>(2.1, 5.3)).r);
}
fn windSway(p : vec2<f32>, t : f32, phase : f32) -> vec2<f32> {
  let d = G.windV.xy; let U = G.windV.z;
  if (U < 0.2) { return vec2<f32>(0.0); }
  let k = min(U / 9.0, 1.4);
  let gust = windGust(p, t);
  let osc = sin(t * (1.6 + 0.6 * phase) + phase * 17.0 + dot(p, d) * 0.08);
  let turb = vnoise4(p * 0.11 + vec2<f32>(t * 0.35, -t * 0.27)).g - 0.5;
  let perp = vec2<f32>(-d.y, d.x);
  return (d * (0.012 + 0.035 * gust + 0.016 * gust * osc) + perp * (0.012 * turb + 0.006 * osc)) * k;
}

// ------------------------------------------------------------------ surf: broken waves (bores) running up the beach
// A function of the signed distance to the waterline (aux.x, + inland): every ~8.5 s a bore arrives with a steep shoreward face and a long
// back that drains; its water runs up the sand (the water pass draws water wherever the surface is above the bed, so the swash and backwash
// come for free) and leaves white water behind it. Fronts are wavy along the shore, and the waves come in sets. None in the lagoon.
// soft = width (m) over which the face is smoothed (vertex spacing or pixel footprint), so the face never aliases.
struct Surf { h : f32, dh : f32, foam : f32 };          // elevation (m), d elevation / d shore distance, white water 0..1
// distance to the nearest cell border of a jittered cell pattern (F2 - F1): foam lace is a network of bubble walls around clear patches
fn cellEdge(p : vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  var d1 = 9.0; var d2 = 9.0;
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    let g = vec2<f32>(f32(x), f32(y));
    let o = g + 0.15 + 0.7 * vec2<f32>(hash21(i + g), hash21(i + g + vec2<f32>(19.19, 7.77))) - f;
    let d = dot(o, o);
    if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
  } }
  return sqrt(d2) - sqrt(d1);
}
const SURF_T : f32 = 8.5;
const SURF_L : f32 = 15.0;
fn surfShape(u : f32, w : f32) -> f32 { let s = fract(u); return pow(1.0 - s, 1.8) * smoothstep(0.0, w, s); }
fn surfSets(p : vec2<f32>, t : f32) -> f32 {           // slow modulation of the height: sets of bigger and smaller waves
  return (0.08 + 0.40 * G.waveA.w) * (0.55 + 0.45 * sin(TAU * t / (SURF_T * 6.5) + vnoise4(p * 0.012 + vec2<f32>(4.1, 7.3)).g * 5.0));
}
fn surfAt(p : vec2<f32>, t : f32, soft : f32) -> Surf {
  var o : Surf; o.h = 0.0; o.dh = 0.0; o.foam = 0.0;
  if (G.misc.w > 0.5 || nearWeight(p) < 0.5) { return o; }
  let sd = auxAt(p).x;
  if (sd < -60.0 || sd > 14.0) { return o; }
  let n = vnoise4(p * 0.012 + vec2<f32>(4.1, 7.3)); let n2 = vnoise4(p * 0.05 + vec2<f32>(1.7, 2.9));
  let si = shoreInfo(p);                                             // exposure: big surf on open shores, little in the lee of islets and headlands
  let A = surfSets(p, t) * smoothstep(-55.0, -18.0, sd) * (1.0 - smoothstep(-3.0, 8.0, sd)) * mix(0.10, 1.15, si.x);
  let n3 = vnoise4(p * 0.0045 + vec2<f32>(9.2, 2.6));
  let x = sd + 9.0 * (n.r - 0.5) + 2.0 * (n2.b - 0.5) + 22.0 * (n3.g - 0.5);   // wavy fronts, and crests that wrap unevenly around points
  let u = (t * SURF_L / SURF_T - x) / SURF_L;                          // the crest is where u is a whole number; it moves inland
  let w = clamp(soft / SURF_L, 0.04, 0.5);
  let e = 0.01;
  // every wave is different: its height (some barely break), and which sections of its crest are breaking. Offshore only a few sections break
  // (over shallower bottom, as peaks); the broken sections spread along the crest as it moves into shallower water, so a wave peels and closes
  // out near the beach instead of arriving as one even ring.
  let k = floor(u);
  let Ak = A * (0.35 + 0.95 * hash21(vec2<f32>(k, 7.31)));
  let inshore = smoothstep(-45.0, -16.0, sd);
  let sec = vnoise4(p * 0.016 + vec2<f32>(k * 3.71, k * 1.93)).r * 0.75 + vnoise4(p * 0.05 + vec2<f32>(k * 2.3, 0.7)).g * 0.25;
  let thr = mix(0.64, 0.02, inshore) + 0.15 * (0.6 - Ak / max(A, 1e-4));
  let broken = smoothstep(thr - 0.10, thr + 0.10, sec);
  o.h = Ak * surfShape(u, w) * (0.55 + 0.45 * broken);               // an unbroken swell is a smoother, lower hump
  o.dh = -Ak * (0.55 + 0.45 * broken) * (surfShape(u + e, w) - surfShape(u - e, w)) / (2.0 * e * SURF_L);
  let s = fract(u);
  // white water: a dense band a couple of metres wide just behind the face, then lace (a network of foam veins with clear water between them)
  // that thins out over the bore's back. The lace drifts slowly so it does not read as a fixed pattern under the moving bore.
  let zone = smoothstep(-48.0, -22.0, sd) * (1.0 - smoothstep(0.0, 8.0, sd)) * smoothstep(0.0, 0.10, A);
  let dense = exp(-s * 7.0) * smoothstep(0.0, w, s) * (0.75 + 0.5 * n2.g);
  let dq = p + vec2<f32>(t * 0.15, -t * 0.11);
  // lace: veins of ridged noise at two scales, with thickness that varies along them and gaps where the foam has torn
  let r1 = 1.0 - abs(vnoise4(dq * vec2<f32>(0.85, 0.55)).r * 2.0 - 1.0);
  let r2 = 1.0 - abs(vnoise4(dq * 2.1 + vec2<f32>(3.3, 1.7)).g * 2.0 - 1.0);
  let vn = vnoise4(dq * 0.6 + vec2<f32>(8.1, 2.2));
  let th = 0.70 + 0.16 * vn.r;                                         // vein thickness varies
  var lace = smoothstep(th, 0.97, max(r1, 0.95 * r2)) * smoothstep(0.22, 0.50, vn.g);
  if (zone > 0.0 && soft < 0.5) {                                      // close up: bubbly texture within the foam
    lace *= 0.75 + 0.35 * smoothstep(0.0, 0.25, cellEdge(dq * 3.5)) * (1.0 - smoothstep(0.2, 0.5, soft));
  }
  lace *= exp(-s * 1.8) * smoothstep(0.0, w, s);
  // white water only behind the broken sections (and less behind a small wave); breaking on a rocky shore throws up more of it
  o.foam = zone * (dense * broken + 0.8 * lace * max(broken, 0.25 * inshore)) * smoothstep(0.0, 0.10, Ak) * (1.0 + 0.8 * si.y);
  return o;
}

// Interactive ripples: (height, dh/dx, dh/dz, agitation), faded to zero at the window boundary.
fn rippleAt(xz : vec2<f32>) -> vec4<f32> {
  let size = G.ripple.z;
  let m = 1.0 - smoothstep(0.36, 0.47, max(abs(xz.x - G.ripple.x), abs(xz.y - G.ripple.y)) / size);
  if (m <= 0.0 || G.ripple.w < 0.05) { return vec4<f32>(0.0); }
  let r = textureSampleLevel(rippleTex, samRep, xz / size, 0.0) * m;
  return vec4<f32>(r.x, r.yz * G.ripple.w, r.w);   // ripple.w = slope gain (0 = disabled)
}

// Caustic modulation on the seabed (mean 1). lod chosen by the caller from the pixel footprint.
fn causticAt(p : vec2<f32>, lod : f32, camDist : f32) -> f32 {
  let uv = (p - G.caustic.xy) / G.caustic.z + 0.5;
  let e = max(abs(uv.x - 0.5), abs(uv.y - 0.5));
  let m = (1.0 - smoothstep(0.38, 0.48, e)) * G.caustic.w * (1.0 - smoothstep(G.causticB.z, G.causticB.w, camDist));
  if (m <= 0.0) { return 1.0; }
  let c = textureSampleLevel(causticTex, samLin, uv, min(lod, G.causticB.y) + 1.0).r;   // level 0 is the 2x supersampled splat
  return mix(1.0, c, m);
}

// ------------------------------------------------------------------ fog / aerial perspective
// Marine haze over the trade-wind sea (visibility ~25-30 km): extinction a little stronger in blue, so distant land fades toward blue-grey.
// The scattered light takes the colour of the low sky in the viewing direction (the sky LUT carries the forward-scattering glow), so haze is
// warmer and brighter toward the sun and bluer away from it. dir = direction from the eye to the point.
fn applyFog(L : vec3<f32>, dist : f32, dir : vec3<f32>) -> vec3<f32> {
  let ext = G.horizon.w * vec3<f32>(1.0, 1.10, 1.38);
  let T = exp(-ext * dist);
  let lowSky = skyRadiance(normalize(vec3<f32>(dir.x, 0.035, dir.z) + vec3<f32>(1e-4, 0.0, 0.0)));
  return L * T + mix(G.horizon.rgb, lowSky, 0.6) * (1.0 - T);
}

// ------------------------------------------------------------------ ray marching against the DEM
// Sun visibility from a ground point (soft, 20 steps).
fn sunShadow(p : vec3<f32>, jitter : f32) -> f32 {
  let L = G.sunDir.xyz;
  var t = 1.5 + jitter; var vis = 1.0;
  for (var i = 0; i < 20; i++) {
    let q = p + L * t;
    // crowns within a few metres are shadowed by crownShade (their real domes); the 4 m stand mean would cast polygonal shadows that close
    let d = q.y - heightAt(q.xz) - meanCanopy(q.xz).x * smoothstep(5.0, 14.0, t);
    vis = min(vis, clamp(d / (0.09 * t + 0.6), 0.0, 1.0));
    if (vis <= 0.001 || q.y > 340.0) { break; }
    t *= 1.34;
  }
  return vis * vis * (3.0 - 2.0 * vis);
}
// Reflected-ray march against terrain above water. Returns (sample position, coverage). Coverage is a SYMMETRIC ramp in the ray's angular
// clearance over the terrain skyline (tan units): 1 when the ray is buried deeper than 'soft', 0 when it clears by more than 'soft', 0.5 at grazing.
// 'soft' is the angular spread of reflections caused by slope variance the caller could not resolve, so distant dark islets smear
// into soft vertical streaks instead of aliasing into binary dashes.
fn marchLand(o : vec3<f32>, r : vec3<f32>, soft : f32) -> vec4<f32> {
  var t = 4.0; var tPrev = 0.5;
  var minClear = 1e9; var tMin = 4.0;
  for (var i = 0; i < 44; i++) {
    let q = o + r * t;
    let h = surfaceAt(q.xz);
    let d = q.y - h;
    let clear = d / t;                       // tan(angular clearance above the terrain) seen from the origin
    if (clear < minClear) { minClear = clear; tMin = t; }
    if (d < 0.0) {
      var a = tPrev; var b = t;
      for (var k = 0; k < 4; k++) {
        let m = 0.5 * (a + b); let qm = o + r * m;
        if (qm.y - surfaceAt(qm.xz) > 0.0) { a = m; } else { b = m; }
      }
      return vec4<f32>(o + r * (0.5 * (a + b)), (1.0 - smoothstep(-soft, soft, clear)) * 0.98);
    }
    if (q.y > 340.0 && r.y > 0.0) { break; }
    tPrev = t;
    t += max(2.5, max(0.13 * t, 0.30 * d));
    if (t > 9000.0) { break; }
  }
  return vec4<f32>(o + r * tMin, (1.0 - smoothstep(-soft, soft, minClear)) * 0.98);
}
// Refracted-ray march from the water surface down to the seabed. Always returns a point (assumed bed at tEnd if none found).
fn marchBed(P : vec3<f32>, T : vec3<f32>, tEnd : f32) -> vec4<f32> {
  let N = 26;
  let dt = tEnd / f32(N);
  var tPrev = 0.0; var gPrev = P.y - heightAt(P.xz);
  for (var i = 1; i <= N; i++) {
    let t = dt * f32(i);
    let q = P + T * t;
    let g = q.y - heightAt(q.xz);
    if (g <= 0.0) {
      var a = tPrev; var b = t;
      let s = gPrev / max(gPrev - g, 1e-5);
      var m0 = tPrev + (t - tPrev) * s;
      let qm0 = P + T * m0;
      if (qm0.y - heightAt(qm0.xz) > 0.0) { a = m0; } else { b = m0; }
      for (var k = 0; k < 4; k++) {
        let m = 0.5 * (a + b); let qm = P + T * m;
        if (qm.y - heightAt(qm.xz) > 0.0) { a = m; } else { b = m; }
      }
      let th = 0.5 * (a + b);
      return vec4<f32>(P + T * th, th);
    }
    tPrev = t; gPrev = g;
  }
  return vec4<f32>(P + T * tEnd, -tEnd);
}
`;
  };
})();
