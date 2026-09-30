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
fn heightFar(p : vec2<f32>) -> f32 {
  return textureSampleLevel(demFarTex, samLin, (p - G.demFar.xy) / G.demFar.zw, 0.0).r;
}
fn heightNear(p : vec2<f32>) -> f32 {
  return textureSampleLevel(demNearTex, samLin, (p - G.demNear.xy) / G.demNear.zw, 0.0).r;
}
fn heightAt(p : vec2<f32>) -> f32 {
  let w = nearWeight(p);
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
const CROWN_OFF : f32 = 16.0;
struct Crown { c : vec2<f32>, H : f32, R : f32, id : f32 };
fn crownAt(p : vec2<f32>) -> Crown {
  var cr : Crown; cr.H = 0.0; cr.R = 3.0; cr.id = 0.0; cr.c = p;
  let t = (p - G.crownA.xy) / G.crownA.z;
  if (t.x < 0.0 || t.y < 0.0 || t.x >= G.crownA.w || t.y >= G.crownB.x) { return cr; }
  let ti = vec2<i32>(floor(t));
  let d = textureLoad(crownTex, ti, 0);
  cr.c = (vec2<f32>(ti) + 0.5) * G.crownA.z + G.crownA.xy + (d.xy - 0.5) * CROWN_OFF;
  cr.H = d.z * G.crownB.z;
  cr.id = d.w;
  cr.R = 3.0 + 2.4 * fract(d.w * 7.31 + 0.17);
  return cr;
}
// tf = local tree fraction (foliage between crowns exists only inside a dense stand, never beside open sand)
fn crownProfile(r : f32, tf : f32) -> f32 {
  let dome = pow(max(1.0 - pow(r, 2.4), 0.0), 0.55);         // flat-topped, steep-sided (umbrella crowns): survives 2 m mesh sampling without turning into cones
  return 0.50 * (1.0 - smoothstep(1.30, 2.1, r)) * smoothstep(0.55, 0.90, tf) + 0.50 * dome;
}
fn crownH(p : vec2<f32>, cr : Crown, tf : f32) -> f32 {
  if (cr.H <= 0.0) { return 0.0; }
  return cr.H * crownProfile(length(p - cr.c) / cr.R, tf);
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
fn meanCanopy(p : vec2<f32>) -> vec2<f32> {
  let uv = (p - G.demNear.xy) / G.demNear.zw;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { return vec2<f32>(0.0); }
  let m = textureSampleLevel(meanTex, samLin, uv, 0.0);
  return vec2<f32>(m.x * G.crownB.z, m.y);
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
  return textureSampleLevel(skyLut, samLin, skyLutUV(dir), 0.0).rgb;
}
fn sunDisc(dir : vec3<f32>) -> vec3<f32> {
  let c = dot(dir, G.sunDir.xyz);
  let a2 = 2.0 * (1.0 - c);
  let r2 = SUN_RADIUS * SUN_RADIUS;
  let edge = 1.0 - smoothstep(r2 * 0.55, r2 * 1.15, a2);
  let mu = sqrt(max(1.0 - a2 / r2, 0.0));
  return G.sunE.rgb / (PI * r2) * edge * (0.62 + 0.38 * mu);
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
fn skyWithClouds(dir : vec3<f32>) -> vec3<f32> {
  let c = clouds(dir);
  return mix(skyRadiance(dir), c.rgb, c.a);
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
fn applyFog(L : vec3<f32>, dist : f32) -> vec3<f32> {
  let ext = G.horizon.w * vec3<f32>(1.0, 1.10, 1.38);
  let T = exp(-ext * dist);
  return L * T + G.horizon.rgb * (1.0 - T);
}

// ------------------------------------------------------------------ ray marching against the DEM
// Sun visibility from a ground point (soft, 20 steps).
fn sunShadow(p : vec3<f32>, jitter : f32) -> f32 {
  let L = G.sunDir.xyz;
  var t = 1.5 + jitter; var vis = 1.0;
  for (var i = 0; i < 20; i++) {
    let q = p + L * t;
    let d = q.y - surfaceAt(q.xz);
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
