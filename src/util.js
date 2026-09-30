// ClearVieques — shared utilities: math, uniform blocks, half-float packing, logging.
// Classic script (no modules) so the demo also runs from file://.
(function () {
  'use strict';
  const CV = (window.CV = window.CV || {});
  CV.D2R = Math.PI / 180;

  // ---------------------------------------------------------------- logging
  CV.log = {
    lines: [],
    onError: null,
    info(...a) { console.log('[CV]', ...a); },
    warn(...a) { console.warn('[CV]', ...a); },
    error(...a) {
      console.error('[CV]', ...a);
      const msg = a.map(x => (x && x.message) || String(x)).join(' ');
      this.lines.push(msg);
      if (this.onError) this.onError(msg);
    },
  };

  // ---------------------------------------------------------------- scalar helpers
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const mix = (a, b, t) => a + (b - a) * t;
  const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  CV.clamp = clamp; CV.mix = mix; CV.smoothstep = smoothstep;

  // ---------------------------------------------------------------- vec3 (plain arrays)
  const v3 = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    len: a => Math.hypot(a[0], a[1], a[2]),
    norm: a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  };
  CV.v3 = v3;

  // ---------------------------------------------------------------- mat4 (column-major, WGSL-compatible)
  const m4 = {
    identity() { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; },
    mul(a, b) { // a * b
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
      return o;
    },
    // Reverse-Z, infinite far plane, right-handed view space (looks down -Z), NDC z in [0,1] (near -> 1, far -> 0).
    perspectiveRevZ(fovY, aspect, near) {
      const f = 1 / Math.tan(fovY / 2), o = new Float32Array(16);
      o[0] = f / aspect; o[5] = f; o[10] = 0; o[11] = -1; o[14] = near; o[15] = 0;
      return o;
    },
    // View matrix from eye + forward + world-up.
    view(eye, fwd, up) {
      const f = v3.norm(fwd), r = v3.norm(v3.cross(f, up)), u = v3.cross(r, f), o = new Float32Array(16);
      o[0] = r[0]; o[4] = r[1]; o[8] = r[2];   o[12] = -v3.dot(r, eye);
      o[1] = u[0]; o[5] = u[1]; o[9] = u[2];   o[13] = -v3.dot(u, eye);
      o[2] = -f[0]; o[6] = -f[1]; o[10] = -f[2]; o[14] = v3.dot(f, eye);
      o[3] = 0; o[7] = 0; o[11] = 0; o[15] = 1;
      return o;
    },
  };
  CV.m4 = m4;

  // Frustum side planes + near plane from a column-major view-projection matrix (WebGPU clip volume: -w<=x,y<=w, 0<=z<=w; reverse-Z is fine, only
  // the near plane differs and it is not used for culling). Planes are unnormalised (sign tests only): [a, b, c, d] with a*x+b*y+c*z+d >= 0 inside.
  CV.frustumPlanes = function (m) {
    const row = (r) => [m[r], m[4 + r], m[8 + r], m[12 + r]];
    const r0 = row(0), r1 = row(1), r3 = row(3);
    const add = (a, b) => a.map((v, i) => v + b[i]), sub = (a, b) => a.map((v, i) => v - b[i]);
    return [add(r3, r0), sub(r3, r0), add(r3, r1), sub(r3, r1)];
  };
  // box = { x0, x1, y0, y1, z0, z1 }; true when the box is (at least partly) inside all planes
  CV.boxInFrustum = function (pl, b) {
    for (let i = 0; i < 4; i++) {
      const q = pl[i];
      const x = q[0] >= 0 ? b.x1 : b.x0, y = q[1] >= 0 ? b.y1 : b.y0, z = q[2] >= 0 ? b.z1 : b.z0;
      if (q[0] * x + q[1] * y + q[2] * z + q[3] < 0) return false;
    }
    return true;
  };

  // ---------------------------------------------------------------- uniform block (vec4 / mat4 fields only => no padding traps)
  CV.UniformBlock = class UniformBlock {
    constructor(fields) {
      this.fields = fields; this.offsets = {}; let off = 0;
      for (const [name, kind] of fields) { this.offsets[name] = off; off += kind === 'mat4' ? 16 : 4; }
      this.data = new Float32Array(off);
      this.byteLength = Math.ceil(off * 4 / 16) * 16;
    }
    wgsl(structName) {
      const body = this.fields.map(([n, k]) => `  ${n} : ${k === 'mat4' ? 'mat4x4<f32>' : 'vec4<f32>'},`).join('\n');
      return `struct ${structName} {\n${body}\n};`;
    }
    set(name, x = 0, y = 0, z = 0, w = 0) {
      const o = this.offsets[name]; if (o === undefined) throw new Error('unknown uniform ' + name);
      const d = this.data; d[o] = x; d[o + 1] = y; d[o + 2] = z; d[o + 3] = w;
    }
    setW(name, w) { this.data[this.offsets[name] + 3] = w; }
    setMat(name, m) { this.data.set(m, this.offsets[name]); }
    get(name, i) { return this.data[this.offsets[name] + i]; }
  };

  // ---------------------------------------------------------------- half floats
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  CV.f32ToF16 = function (val) {
    f32[0] = val; const x = u32[0];
    const sign = (x >>> 16) & 0x8000; let exp = ((x >>> 23) & 0xff) - 127 + 15; let mant = x & 0x7fffff;
    if (exp >= 31) return sign | 0x7c00 | (mant && exp === 143 ? 0x200 : 0); // inf/nan
    if (exp <= 0) { // subnormal / zero
      if (exp < -10) return sign;
      mant |= 0x800000; const shift = 14 - exp; let h = mant >> shift;
      if ((mant >> (shift - 1)) & 1) h++;
      return sign | h;
    }
    let h = sign | (exp << 10) | (mant >> 13);
    if (mant & 0x1000) h++; // round to nearest
    return h;
  };
  CV.toHalf = function (arr) { const o = new Uint16Array(arr.length); for (let i = 0; i < arr.length; i++) o[i] = CV.f32ToF16(arr[i]); return o; };

  // ---------------------------------------------------------------- misc
  CV.b64ToBytes = function (b64) {
    const bin = atob(b64), n = bin.length, out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
    return out;
  };
  CV.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0; let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  CV.nextPow2 = n => 1 << Math.ceil(Math.log2(n));
  CV.mipCount = (w, h) => Math.floor(Math.log2(Math.max(w, h))) + 1;
})();
