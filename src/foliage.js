// ClearVieques — foliage micro-structure texture. Tree crowns are shaded with a tileable, mip-mapped RGBA8 texture of "leaf clusters": a height field of
// rounded clusters piled in clusters of clusters (three scales of domain-warped, variable-size Voronoi domes with creases between them), baked once on
// the CPU. One texel carries everything the terrain shader needs per octave, so a crown pixel costs one fetch per octave and the mip chain does the
// level-of-detail fade for free (a far-away crown averages to zero slope, mean occlusion and mean tint):
//   r, g : slope of the cluster surface (d height / d position in the texture's two axes), 0.5 = flat, normalised so the 99.5th percentile is +-1
//   b    : occlusion / exposure, rank-transformed (uniform 0..1): low in the creases and hollows between clusters, high on top of them
//   a    : random tint per mid-scale cluster (uniform 0..1)
(function () {
  'use strict';
  const CV = window.CV;
  const N = 256, CELLS = [6, 14, 30], WEIGHT = [1.0, 0.50, 0.26];

  function build() {
    const rnd = CV.mulberry32(9071);
    // periodic value noise (domain warp)
    const G = 16, grid = new Float32Array(G * G * 2); for (let i = 0; i < grid.length; i++) grid[i] = rnd();
    const smooth = t => t * t * (3 - 2 * t);
    const warp = (u, v, k) => {   // u, v in tile units [0,1)
      const x = u * G, y = v * G, i = Math.floor(x), j = Math.floor(y), fx = smooth(x - i), fy = smooth(y - j);
      const at = (a, b) => grid[(((b % G) + G) % G * G + ((a % G) + G) % G) * 2 + k];
      return (at(i, j) * (1 - fx) + at(i + 1, j) * fx) * (1 - fy) + (at(i, j + 1) * (1 - fx) + at(i + 1, j + 1) * fx) * fy - 0.5;
    };
    const H = new Float32Array(N * N), ID = new Float32Array(N * N);
    CELLS.forEach((cells, li) => {
      const fx = new Float32Array(cells * cells), fy = new Float32Array(cells * cells), fr = new Float32Array(cells * cells), fid = new Float32Array(cells * cells);
      for (let i = 0; i < cells * cells; i++) { fx[i] = 0.12 + 0.76 * rnd(); fy[i] = 0.12 + 0.76 * rnd(); fr[i] = 0.50 + 0.34 * rnd(); fid[i] = rnd(); }
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const u = (x + 0.5) / N, v = (y + 0.5) / N;
        const qx = (u + warp(u, v, 0) * 0.55 / cells * 1.8) * cells, qy = (v + warp(u, v, 1) * 0.55 / cells * 1.8) * cells;   // warped lookup: irregular clusters
        const ix = Math.floor(qx), iy = Math.floor(qy), px = qx - ix, py = qy - iy;
        let d1 = 9, d2 = 9, id1 = 0, ci = 0;
        for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
          const cx = ((ix + a) % cells + cells) % cells, cy = ((iy + b) % cells + cells) % cells, c = cy * cells + cx;
          const dx = a + fx[c] - px, dy = b + fy[c] - py, d = Math.hypot(dx, dy) / fr[c];
          if (d < d1) { d2 = d1; d1 = d; id1 = fid[c]; ci = c; } else if (d < d2) d2 = d;
        }
        const w = Math.max(1 - Math.min(d1, 1) ** 2, 0.05), dome = w ** 0.7;
        const t = Math.min(Math.max((d2 - d1) / 0.40, 0), 1), crease = t * t * (3 - 2 * t);
        H[y * N + x] += WEIGHT[li] * (dome - 0.55 * (1 - crease));
        if (li === 1) ID[y * N + x] = id1;
      }
    });
    // slope (periodic central differences of a lightly smoothed height field)
    const blur = (src, r) => {   // separable box blur, periodic
      const tmp = new Float32Array(N * N), out = new Float32Array(N * N), n = 2 * r + 1;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let s = 0; for (let k = -r; k <= r; k++) s += src[y * N + (x + k + N) % N]; tmp[y * N + x] = s / n; }
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let s = 0; for (let k = -r; k <= r; k++) s += tmp[((y + k + N) % N) * N + x]; out[y * N + x] = s / n; }
      return out;
    };
    const Hs = blur(H, 1), gx = new Float32Array(N * N), gy = new Float32Array(N * N), mag = [];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x;
      gx[i] = (Hs[y * N + (x + 1) % N] - Hs[y * N + (x + N - 1) % N]) * 0.5 * N;
      gy[i] = (Hs[((y + 1) % N) * N + x] - Hs[((y + N - 1) % N) * N + x]) * 0.5 * N;
      mag.push(Math.max(Math.abs(gx[i]), Math.abs(gy[i])));
    }
    mag.sort((a, b) => a - b);
    const gScale = mag[Math.floor(mag.length * 0.995)] || 1;
    // occlusion: hollows (below the local mean) are dark, tops are bright; rank-transformed to a uniform distribution so its mean is exactly 0.5
    const Hb = blur(H, 14), raw = new Float32Array(N * N), order = new Uint32Array(N * N);
    for (let i = 0; i < raw.length; i++) { raw[i] = H[i] - 0.55 * Hb[i]; order[i] = i; }
    order.sort((a, b) => raw[a] - raw[b]);
    const rank = new Float32Array(N * N); for (let k = 0; k < order.length; k++) rank[order[k]] = (k + 0.5) / order.length;

    const level0 = new Uint8Array(N * N * 4);
    for (let i = 0; i < N * N; i++) {
      level0[i * 4] = Math.round(255 * Math.min(Math.max(0.5 + 0.5 * gx[i] / gScale, 0), 1));
      level0[i * 4 + 1] = Math.round(255 * Math.min(Math.max(0.5 + 0.5 * gy[i] / gScale, 0), 1));
      level0[i * 4 + 2] = Math.round(255 * rank[i]);
      level0[i * 4 + 3] = Math.round(255 * ID[i]);
    }
    return level0;
  }

  // box-filtered mip chain (gradients average toward flat, occlusion and tint toward their means)
  function mips(level0) {
    const out = [level0]; let w = N, src = level0;
    while (w > 1) {
      const nw = w >> 1, dst = new Uint8Array(nw * nw * 4);
      for (let y = 0; y < nw; y++) for (let x = 0; x < nw; x++) for (let c = 0; c < 4; c++) {
        const a = src[((2 * y) * w + 2 * x) * 4 + c], b = src[((2 * y) * w + 2 * x + 1) * 4 + c], d = src[((2 * y + 1) * w + 2 * x) * 4 + c], e = src[((2 * y + 1) * w + 2 * x + 1) * 4 + c];
        dst[(y * nw + x) * 4 + c] = (a + b + d + e + 2) >> 2;
      }
      out.push(dst); src = dst; w = nw;
    }
    return out;
  }

  CV.makeFoliageTexture = function (gpu) {
    const chain = mips(build());
    const tex = gpu.device.createTexture({ label: 'foliage', size: [N, N], format: 'rgba8unorm', mipLevelCount: chain.length,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    chain.forEach((data, level) => { const w = N >> level; gpu.queue.writeTexture({ texture: tex, mipLevel: level }, data, { bytesPerRow: w * 4 }, [w, w]); });
    return tex;
  };
})();
