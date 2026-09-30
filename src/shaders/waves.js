// ClearVieques — WGSL compute kernels for the spectral wave cascades.
//   initSpectrum : Gaussian h0(k) from wind-sea (JONSWAP) + swell, band-limited per cascade
//   evolve       : h(k,t) and the 8 derived real fields, packed as 4 complex spectra (2 real fields per complex IFFT)
//   fftRows/Cols : in-workgroup Stockham radix-2 inverse FFT (256 points, 128 threads)
//   assemble     : displacement (choppy), slopes, Jacobian and persistent foam into cascade texture arrays
(function () {
  'use strict';
  const CV = window.CV;
  CV.wgsl = CV.wgsl || {};
  const N = 256;

  const common = /* wgsl */`
const N : u32 = ${N}u;
const GRAV : f32 = 9.81;
const TAU : f32 = 6.28318530718;

struct WaveU {
  a : vec4<f32>,       // t (wrapped), choppy, spectrum depth, omega0
  b : vec4<f32>,       // foam gain, foam decay (per frame), foam threshold, choppy sign
  casc0 : vec4<f32>,   // L, kLo, kHi, seed
  casc1 : vec4<f32>,
  casc2 : vec4<f32>,
  wind : vec4<f32>,    // alpha, wp, gamma, dir (rad, direction of travel: atan2(dz,dx))
  windB : vec4<f32>,   // spread s at the peak, spread s at high k, peak wavenumber kp, -
  swell : vec4<f32>,
  swellB : vec4<f32>,
};
fn cascadeOf(u : WaveU, c : u32) -> vec4<f32> {
  if (c == 0u) { return u.casc0; }
  if (c == 1u) { return u.casc1; }
  return u.casc2;
}
fn cmul(a : vec2<f32>, b : vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn mulI(v : vec2<f32>) -> vec2<f32> { return vec2<f32>(-v.y, v.x); }
`;

  CV.wgsl.waveInit = common + /* wgsl */`
@group(0) @binding(0) var<uniform> U : WaveU;
@group(0) @binding(1) var<storage, read_write> h0buf : array<vec2<f32>>;

fn pcg(v : u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand01(v : u32) -> f32 { return f32(pcg(v) >> 8u) / 16777216.0; }

fn dispersion(k : f32, h : f32) -> f32 {
  let kh = min(k * h, 40.0);
  return sqrt(GRAV * k * tanh(kh) * (1.0 + (k / 363.0) * (k / 363.0)));
}
fn groupVel(k : f32, h : f32) -> f32 { // d omega / dk (numerical, robust)
  let e = k * 0.002 + 1e-5;
  return (dispersion(k + e, h) - dispersion(max(k - e, 1e-6), h)) / (2.0 * e);
}
fn jonswap(w : f32, alpha : f32, wp : f32, gamma : f32) -> f32 {
  if (w < 1e-3) { return 0.0; }
  let sigma = select(0.09, 0.07, w <= wp);
  let r = exp(-(w - wp) * (w - wp) / (2.0 * sigma * sigma * wp * wp));
  return alpha * GRAV * GRAV * pow(w, -5.0) * exp(-1.25 * pow(wp / w, 4.0)) * pow(gamma, r);
}
fn dirSpread(theta : f32, dir : f32, s : f32, norm : f32) -> f32 {
  let d = theta - dir;
  let dd = atan2(sin(d), cos(d));
  return norm * pow(max(cos(0.5 * dd), 0.0), 2.0 * s);
}
// Wind-sea spreading widens with wavenumber (Elfouhaily et al. 1997): long waves stay wind-aligned, wavelets a few times shorter than the
// peak are nearly isotropic. s falls from windB.x at k <= 2 kp to windB.y at k >= 16 kp; the cos^2s lobe is normalised by Gamma(s+1)/(2 sqrt(pi) Gamma(s+1/2)) ~ sqrt(s+1/4)/(2 sqrt(pi)).
fn windSpread(theta : f32, dir : f32, k : f32) -> f32 {
  let t = clamp((log2(max(k, 1e-3) / max(U.windB.z, 1e-3)) - 1.0) / 3.0, 0.0, 1.0);
  let s = mix(U.windB.x, U.windB.y, t * t * (3.0 - 2.0 * t));
  return dirSpread(theta, dir, s, sqrt(s + 0.25) * 0.28209479);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let x = id.x; let y = id.y; let c = id.z;
  if (x >= N || y >= N) { return; }
  let cc = cascadeOf(U, c);
  let L = cc.x;
  let nx = select(i32(x), i32(x) - i32(N), x >= N / 2u);
  let nz = select(i32(y), i32(y) - i32(N), y >= N / 2u);
  let dk = TAU / L;
  let kv = vec2<f32>(f32(nx), f32(nz)) * dk;
  let k = length(kv);
  var out = vec2<f32>(0.0);
  if (k > 1e-4 && nx != -i32(N / 2u) && nz != -i32(N / 2u)) {
    let theta = atan2(kv.y, kv.x);
    let w = dispersion(k, U.a.z);
    let cg = groupVel(k, U.a.z);
    var psi = 0.0;
    psi += jonswap(w, U.wind.x, U.wind.y, U.wind.z) * windSpread(theta, U.wind.w, k);
    psi += jonswap(w, U.swell.x, U.swell.y, U.swell.z) * dirSpread(theta, U.swell.w, U.swellB.x, U.swellB.y);
    psi *= cg / k;
    // smooth band limits between cascades + Nyquist guard
    var win = 1.0;
    if (cc.y > 0.0) { win *= smoothstep(cc.y * 0.85, cc.y * 1.15, k); }
    if (cc.z < 1e8) { win *= 1.0 - smoothstep(cc.z * 0.85, cc.z * 1.15, k); }
    let kn = 3.14159265 * f32(N) / L;
    win *= 1.0 - smoothstep(0.72 * kn, 0.94 * kn, max(abs(kv.x), abs(kv.y)));
    psi *= win;
    // complex Gaussian
    let seed = pcg(x + pcg(y * 7919u + pcg(c * 104729u + u32(cc.w))));
    let u1 = max(rand01(seed), 1e-7); let u2 = rand01(seed ^ 0x9E3779B9u);
    let r = sqrt(-2.0 * log(u1)); let ph = TAU * u2;
    let xi = vec2<f32>(r * cos(ph), r * sin(ph));
    out = xi * (0.5 * dk * sqrt(max(psi, 0.0)));
  }
  h0buf[c * N * N + y * N + x] = out;
}
`;

  CV.wgsl.waveEvolve = common + /* wgsl */`
@group(0) @binding(0) var<uniform> U : WaveU;
@group(0) @binding(1) var<storage, read> h0buf : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> outA : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outB : array<vec4<f32>>;

fn dispersion(k : f32, h : f32) -> f32 {
  let kh = min(k * h, 40.0);
  return sqrt(GRAV * k * tanh(kh) * (1.0 + (k / 363.0) * (k / 363.0)));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let x = id.x; let y = id.y; let c = id.z;
  if (x >= N || y >= N) { return; }
  let cc = cascadeOf(U, c);
  let nx = select(i32(x), i32(x) - i32(N), x >= N / 2u);
  let nz = select(i32(y), i32(y) - i32(N), y >= N / 2u);
  let dk = TAU / cc.x;
  let kx = f32(nx) * dk; let kz = f32(nz) * dk;
  let kl = sqrt(kx * kx + kz * kz);
  let base = c * N * N;
  let i0 = base + y * N + x;
  if (kl < 1e-5) { outA[i0] = vec4<f32>(0.0); outB[i0] = vec4<f32>(0.0); return; }
  let xm = (N - x) & (N - 1u); let ym = (N - y) & (N - 1u);
  let a = h0buf[i0]; let b = h0buf[base + ym * N + xm];
  let w0 = U.a.w;
  let wq = round(dispersion(kl, U.a.z) / w0) * w0;
  let ph = wq * U.a.x;
  let e = vec2<f32>(cos(ph), sin(ph));
  let H = cmul(a, e) + cmul(vec2<f32>(b.x, -b.y), vec2<f32>(e.x, -e.y));
  let kn = vec2<f32>(kx, kz) / kl;
  let Dx = vec2<f32>(H.y * kn.x, -H.x * kn.x);   // -i kx/|k| H
  let Dz = vec2<f32>(H.y * kn.y, -H.x * kn.y);
  let Sx = vec2<f32>(-kx * H.y, kx * H.x);       //  i kx H
  let Sz = vec2<f32>(-kz * H.y, kz * H.x);
  let Dxx = H * (kx * kx / kl); let Dzz = H * (kz * kz / kl); let Dxz = H * (kx * kz / kl);
  outA[i0] = vec4<f32>(H + mulI(Dx), Dz + mulI(Sx));
  outB[i0] = vec4<f32>(Sz + mulI(Dxx), Dzz + mulI(Dxz));
}
`;

  // Stockham radix-2 autosort FFT (inverse), 256 points, 128 threads, two vec4 buffers (= 4 complex signals) per dispatch.
  const fftCore = /* wgsl */`
const N : u32 = ${N}u;
var<workgroup> pa : array<vec4<f32>, ${N}>;
var<workgroup> pb : array<vec4<f32>, ${N}>;
@group(0) @binding(0) var<storage, read_write> bufA : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> bufB : array<vec4<f32>>;

fn cmul4(v : vec4<f32>, w : vec2<f32>) -> vec4<f32> {
  return vec4<f32>(v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x, v.z * w.x - v.w * w.y, v.z * w.y + v.w * w.x);
}
fn fft256(t : u32) {
  var p = 1u;
  for (var s = 0u; s < 8u; s++) {
    let k = t & (p - 1u);
    let ang = 3.14159265359 * f32(k) / f32(p);
    let tw = vec2<f32>(cos(ang), sin(ang));
    var u0 : vec4<f32>; var u1 : vec4<f32>;
    if ((s & 1u) == 0u) { u0 = pa[t]; u1 = cmul4(pa[t + 128u], tw); } else { u0 = pb[t]; u1 = cmul4(pb[t + 128u], tw); }
    let j = ((t - k) << 1u) + k;
    if ((s & 1u) == 0u) { pb[j] = u0 + u1; pb[j + p] = u0 - u1; } else { pa[j] = u0 + u1; pa[j + p] = u0 - u1; }
    workgroupBarrier();
    p = p << 1u;
  }
}
`;
  CV.wgsl.waveFftRows = fftCore + /* wgsl */`
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) t : u32, @builtin(workgroup_id) wid : vec3<u32>) {
  let base = wid.y * N * N + wid.x * N;
  pa[t] = bufA[base + t]; pa[t + 128u] = bufA[base + t + 128u];
  workgroupBarrier();
  fft256(t);
  bufA[base + t] = pa[t]; bufA[base + t + 128u] = pa[t + 128u];
  workgroupBarrier();
  pa[t] = bufB[base + t]; pa[t + 128u] = bufB[base + t + 128u];
  workgroupBarrier();
  fft256(t);
  bufB[base + t] = pa[t]; bufB[base + t + 128u] = pa[t + 128u];
}
`;
  CV.wgsl.waveFftCols = fftCore + /* wgsl */`
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) t : u32, @builtin(workgroup_id) wid : vec3<u32>) {
  let base = wid.y * N * N + wid.x;
  pa[t] = bufA[base + t * N]; pa[t + 128u] = bufA[base + (t + 128u) * N];
  workgroupBarrier();
  fft256(t);
  bufA[base + t * N] = pa[t]; bufA[base + (t + 128u) * N] = pa[t + 128u];
  workgroupBarrier();
  pa[t] = bufB[base + t * N]; pa[t + 128u] = bufB[base + (t + 128u) * N];
  workgroupBarrier();
  fft256(t);
  bufB[base + t * N] = pa[t]; bufB[base + (t + 128u) * N] = pa[t + 128u];
}
`;

  CV.wgsl.waveAssemble = common + /* wgsl */`
@group(0) @binding(0) var<uniform> U : WaveU;
@group(0) @binding(1) var<storage, read> bufA : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bufB : array<vec4<f32>>;
@group(0) @binding(3) var prevSlope : texture_2d_array<f32>;
@group(0) @binding(4) var outDisp : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var outSlope : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let x = id.x; let y = id.y; let c = id.z;
  if (x >= N || y >= N) { return; }
  let i = c * N * N + y * N + x;
  let a = bufA[i]; let b = bufB[i];
  let h = a.x; let Dx = a.y; let Dz = a.z; let Sx = a.w;
  let Sz = b.x; let Dxx = b.y; let Dzz = b.z; let Dxz = b.w;
  let lam = U.a.y * U.b.w;                       // choppy * sign (crest sharpening: x' = x - lam*D)
  let J = (1.0 - lam * Dxx) * (1.0 - lam * Dzz) - lam * lam * Dxz * Dxz;
  let pos = vec2<i32>(i32(x), i32(y));
  let prevFoam = textureLoad(prevSlope, pos, i32(c), 0).w;
  let src = select(0.0, clamp((U.b.z - J) * U.b.x, 0.0, 1.0), c < 2u);   // only gravity waves can break; capillary ripples never foam
  let foam = max(prevFoam * U.b.y, src);
  textureStore(outDisp, pos, i32(c), vec4<f32>(-lam * Dx, h, -lam * Dz, 0.0));
  textureStore(outSlope, pos, i32(c), vec4<f32>(Sx, Sz, J, foam));
}
`;
})();
