// ClearVieques — real topo-bathy terrain (NOAA CUDEM / NGS lidar subsets, see data/ and tools/).
// Frame: x = east (m), y = up (m above mean sea level datum), z = SOUTH (m). North is -z.
// Grids are row-major, row 0 = northernmost; cell (i,j) centre = (x0 + (i+.5)dx, z0 + (j+.5)dz).
(function () {
  'use strict';
  const CV = window.CV;

  function decodeGrid(g) {
    const bytes = CV.b64ToBytes(g.b64);
    const i16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
    if (i16.length !== g.nx * g.nz) throw new Error(`terrain grid size mismatch: ${i16.length} vs ${g.nx}x${g.nz}`);
    const h = new Float32Array(i16.length);
    for (let i = 0; i < h.length; i++) h[i] = i16[i] * 0.01; // centimetres -> metres
    return { nx: g.nx, nz: g.nz, dx: g.dx, dz: g.dz || g.dx, x0: g.x0, z0: g.z0, h };
  }

  function distanceTransform(src, nx, nz, cell) { // src: Uint8 (1 = source) -> Float32 distance in metres (0 on sources)
    const INF = 1e9, d = new Float32Array(nx * nz), dg = cell * Math.SQRT2;
    for (let k = 0; k < d.length; k++) d[k] = src[k] ? 0 : INF;
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i; let v = d[k];
      if (i > 0) v = Math.min(v, d[k - 1] + cell);
      if (j > 0) { v = Math.min(v, d[k - nx] + cell); if (i > 0) v = Math.min(v, d[k - nx - 1] + dg); if (i < nx - 1) v = Math.min(v, d[k - nx + 1] + dg); }
      d[k] = v;
    }
    for (let j = nz - 1; j >= 0; j--) for (let i = nx - 1; i >= 0; i--) {
      const k = j * nx + i; let v = d[k];
      if (i < nx - 1) v = Math.min(v, d[k + 1] + cell);
      if (j < nz - 1) { v = Math.min(v, d[k + nx] + cell); if (i < nx - 1) v = Math.min(v, d[k + nx + 1] + dg); if (i > 0) v = Math.min(v, d[k + nx - 1] + dg); }
      d[k] = v;
    }
    return d;
  }

  function boxBlur(src, nx, nz, r) { // separable box blur with clamped edges
    const tmp = new Float32Array(src.length), out = new Float32Array(src.length), n = 2 * r + 1;
    for (let j = 0; j < nz; j++) {
      let s = 0; const row = j * nx;
      for (let i = -r; i <= r; i++) s += src[row + Math.min(nx - 1, Math.max(0, i))];
      for (let i = 0; i < nx; i++) {
        tmp[row + i] = s / n;
        s += src[row + Math.min(nx - 1, i + r + 1)] - src[row + Math.max(0, i - r)];
      }
    }
    for (let i = 0; i < nx; i++) {
      let s = 0;
      for (let j = -r; j <= r; j++) s += tmp[Math.min(nz - 1, Math.max(0, j)) * nx + i];
      for (let j = 0; j < nz; j++) {
        out[j * nx + i] = s / n;
        s += tmp[Math.min(nz - 1, j + r + 1) * nx + i] - tmp[Math.max(0, j - r) * nx + i];
      }
    }
    return out;
  }

  CV.Terrain = class Terrain {
    constructor(gpu, siteId) {
      const T = window.CV_TERRAIN && window.CV_TERRAIN[siteId];
      if (!T) throw new Error(`Terrain data for "${siteId}" not found. Run tools/build_terrain.py + tools/pack_terrain.py (see README).`);
      this.gpu = gpu; this.siteId = siteId; this.meta = T.meta || {}; this.osm = T.osm || null;
      this.near = decodeGrid(T.near); this.far = decodeGrid(T.far);
      const n = this.near;
      this.nearRect = { x0: n.x0, z0: n.z0, x1: n.x0 + n.nx * n.dx, z1: n.z0 + n.nz * n.dz };
      this.blend = 160; // m: near->far height cross-fade width inside the near window
      this.seaLevel = 0;
      this.computeAux();
      this.upload();
    }

    // ------------------------------------------------------------------ CPU height queries (mirror the WGSL heightAt)
    sampleGrid(g, x, z) {
      const u = (x - g.x0) / g.dx - 0.5, v = (z - g.z0) / g.dz - 0.5;
      const i0 = Math.floor(u), j0 = Math.floor(v), fu = u - i0, fv = v - j0, nx = g.nx, nz = g.nz;
      const c = (i, j) => g.h[Math.min(nz - 1, Math.max(0, j)) * nx + Math.min(nx - 1, Math.max(0, i))];
      return (c(i0, j0) * (1 - fu) + c(i0 + 1, j0) * fu) * (1 - fv) + (c(i0, j0 + 1) * (1 - fu) + c(i0 + 1, j0 + 1) * fu) * fv;
    }
    nearWeight(x, z) {
      const r = this.nearRect, d = Math.min(x - r.x0, r.x1 - x, z - r.z0, r.z1 - z);
      return CV.smoothstep(0, this.blend, d);
    }
    heightAt(x, z) {
      const w = this.nearWeight(x, z), hf = this.sampleGrid(this.far, x, z);
      return w > 0 ? CV.mix(hf, this.sampleGrid(this.near, x, z), w) : hf;
    }

    // ------------------------------------------------------------------ auxiliary fields on the near grid
    computeAux() {
      const g = this.near, { nx, nz } = g, N = nx * nz;
      const land = new Uint8Array(N), water = new Uint8Array(N);
      for (let k = 0; k < N; k++) { const isLand = g.h[k] > 0; land[k] = isLand ? 1 : 0; water[k] = isLand ? 0 : 1; }
      const dWater = distanceTransform(water, nx, nz, g.dx); // for land cells: distance to nearest water
      const dLand = distanceTransform(land, nx, nz, g.dx);   // for water cells: distance to nearest land
      const aux = new Float32Array(N * 4);
      const blur = boxBlur(g.h, nx, nz, 5);
      // sparse OSM tracks -> distance field (metres, capped)
      const trackD = new Float32Array(N).fill(48);
      const tracks = (this.osm && (this.osm.tracks || this.osm.roads || this.osm.ways)) || [];
      for (const w of tracks) {
        const pts = w.pts || w.points || w; if (!Array.isArray(pts) || pts.length < 2) continue;
        for (let s = 0; s < pts.length - 1; s++) {
          const ax = pts[s][0], az = pts[s][1], bx = pts[s + 1][0], bz = pts[s + 1][1];
          const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - 48 - g.x0) / g.dx)), i1 = Math.min(nx - 1, Math.ceil((Math.max(ax, bx) + 48 - g.x0) / g.dx));
          const j0 = Math.max(0, Math.floor((Math.min(az, bz) - 48 - g.z0) / g.dz)), j1 = Math.min(nz - 1, Math.ceil((Math.max(az, bz) + 48 - g.z0) / g.dz));
          const ex = bx - ax, ez = bz - az, l2 = ex * ex + ez * ez || 1;
          for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
            const px = g.x0 + (i + 0.5) * g.dx, pz = g.z0 + (j + 0.5) * g.dz;
            const t = Math.max(0, Math.min(1, ((px - ax) * ex + (pz - az) * ez) / l2));
            const d = Math.hypot(px - (ax + ex * t), pz - (az + ez * t)), k = j * nx + i;
            if (d < trackD[k]) trackD[k] = d;
          }
        }
      }
      for (let k = 0; k < N; k++) {
        aux[k * 4] = land[k] ? dWater[k] : -dLand[k]; // signed distance to the (0 m) waterline, + inland
        aux[k * 4 + 1] = blur[k] - g.h[k];             // >0 concave (occluded), <0 convex
        aux[k * 4 + 2] = trackD[k];
        aux[k * 4 + 3] = 0;
      }
      this.aux = aux;
    }

    // ------------------------------------------------------------------ GPU
    upload() {
      const { device, queue } = this.gpu, gpu = this.gpu, f16 = gpu.fmtDem === 'r16float';
      const mk = (g, label) => {
        const tex = device.createTexture({ label, size: [g.nx, g.nz], format: gpu.fmtDem, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        if (f16) queue.writeTexture({ texture: tex }, CV.toHalf(g.h), { bytesPerRow: g.nx * 2 }, [g.nx, g.nz]);
        else queue.writeTexture({ texture: tex }, g.h, { bytesPerRow: g.nx * 4 }, [g.nx, g.nz]);
        return tex;
      };
      this.demNear = mk(this.near, 'demNear');
      this.demFar = mk(this.far, 'demFar');
      const n = this.near;
      this.auxTex = device.createTexture({ label: 'terrainAux', size: [n.nx, n.nz], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      queue.writeTexture({ texture: this.auxTex }, CV.toHalf(this.aux), { bytesPerRow: n.nx * 8 }, [n.nx, n.nz]);

      // Tree-canopy maps (filled by CV.Canopy.bake): crowns at 2 m over the near window, stand mean on the near-DEM grid.
      const cTexel = 2.0, cnx = Math.round(n.nx * n.dx / cTexel), cnz = Math.round(n.nz * n.dz / cTexel);
      this.crown = { x0: n.x0, z0: n.z0, texel: cTexel, nx: cnx, nz: cnz, maxH: 16 };
      const cu = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
      this.crownTex = device.createTexture({ label: 'crownMap', size: [cnx, cnz], format: 'rgba8unorm', usage: cu });
      this.meanTex = device.createTexture({ label: 'canopyMean', size: [n.nx, n.nz], format: 'rgba8unorm', usage: cu });
      this.canopyBaked = false;

      // Grid meshes (vertex positions are derived from the vertex id + DEM texture in the vertex shader).
      // FAR: 20 m cells, one index buffer grouped into tiles (each with a bounding box) so tiles that are off-screen or under water can be skipped.
      const TILE = 32;
      const buildFar = (g) => {
        const cellsX = g.nx - 1, cellsZ = g.nz - 1, tilesX = Math.ceil(cellsX / TILE), tilesZ = Math.ceil(cellsZ / TILE);
        const r = this.nearRect, inner = this.blend;
        let count = 0; const arr = new Uint32Array(cellsX * cellsZ * 6), tiles = [];
        for (let tj = 0; tj < tilesZ; tj++) for (let ti = 0; ti < tilesX; ti++) {
          const first = count, i0 = ti * TILE, j0 = tj * TILE, i1 = Math.min(cellsX, i0 + TILE), j1 = Math.min(cellsZ, j0 + TILE);
          for (let j = j0; j < j1; j++) for (let i = i0; i < i1; i++) {
            // cell fully inside the fully-blended interior of the near window -> drawn by the near mesh
            const xa = g.x0 + (i + 0.5) * g.dx, xb = g.x0 + (i + 1.5) * g.dx, za = g.z0 + (j + 0.5) * g.dz, zb = g.z0 + (j + 1.5) * g.dz;
            if (xa > r.x0 + inner && xb < r.x1 - inner && za > r.z0 + inner && zb < r.z1 - inner) continue;
            const a = j * g.nx + i, b = a + 1, c = a + g.nx, d = c + 1;
            arr[count++] = a; arr[count++] = c; arr[count++] = b; arr[count++] = b; arr[count++] = c; arr[count++] = d;
          }
          if (count === first) continue;
          let hMin = 1e9, hMax = -1e9;   // vertex heights of the tile (the far mesh also samples the near grid inside the blend zone: small margin)
          for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) { const h = g.h[j * g.nx + i]; if (h < hMin) hMin = h; if (h > hMax) hMax = h; }
          tiles.push({ first, count: count - first, x0: g.x0 + (i0 + 0.5) * g.dx, x1: g.x0 + (i1 + 0.5) * g.dx, z0: g.z0 + (j0 + 0.5) * g.dz, z1: g.z0 + (j1 + 0.5) * g.dz, y0: hMin - 6, y1: hMax + 6 });
        }
        const buf = device.createBuffer({ label: 'terrainIdxFar', size: count * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        queue.writeBuffer(buf, 0, arr.buffer, 0, count * 4);
        return { buf, count, tiles, grid: { x0: g.x0, z0: g.z0, dx: g.dx, dz: g.dz, nx: g.nx, nz: g.nz } };
      };
      // NEAR: 2 m cells (half the DEM spacing) so tree crowns are several vertices wide. Every tile draws the same 32x32-cell index pattern; the tile is
      // selected with baseVertex, so the index buffer is 24 KB instead of tens of MB. The DEM is bilinearly interpolated between its 4 m samples.
      const buildNear = () => {
        const cell = 2.0, nx = Math.round(n.nx * n.dx / cell), nz = Math.round(n.nz * n.dz / cell), T = 32;
        const pat = new Uint32Array(T * T * 6); let k = 0;
        for (let j = 0; j < T; j++) for (let i = 0; i < T; i++) {
          const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
          pat[k++] = a; pat[k++] = c; pat[k++] = b; pat[k++] = b; pat[k++] = c; pat[k++] = d;
        }
        const idx = device.createBuffer({ label: 'terrainIdxNear', size: pat.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        queue.writeBuffer(idx, 0, pat);
        const tiles = [], tx = Math.ceil((nx - 1) / T), tz = Math.ceil((nz - 1) / T), canopyMax = this.crown.maxH;
        for (let tj = 0; tj < tz; tj++) for (let ti = 0; ti < tx; ti++) {
          const i0 = ti * T, j0 = tj * T;
          if (i0 + T > nx - 1 || j0 + T > nz - 1) continue;          // partial edge tiles are dropped (window edge, fully inside the far mesh's blend band)
          const x0 = n.x0 + (i0 + 0.5) * cell, x1 = x0 + T * cell, z0 = n.z0 + (j0 + 0.5) * cell, z1 = z0 + T * cell;
          const ga = Math.max(0, Math.floor((x0 - n.x0) / n.dx) - 1), gb = Math.min(n.nx - 1, Math.ceil((x1 - n.x0) / n.dx) + 1);
          const gc = Math.max(0, Math.floor((z0 - n.z0) / n.dz) - 1), gd = Math.min(n.nz - 1, Math.ceil((z1 - n.z0) / n.dz) + 1);
          let hMin = 1e9, hMax = -1e9;
          for (let j = gc; j <= gd; j++) for (let i = ga; i <= gb; i++) { const h = n.h[j * n.nx + i]; if (h < hMin) hMin = h; if (h > hMax) hMax = h; }
          tiles.push({ base: j0 * nx + i0, count: T * T * 6, x0, x1, z0, z1, y0: hMin - 6, y1: hMax + canopyMax + 2 });
        }
        return { idx, tiles, grid: { x0: n.x0, z0: n.z0, dx: cell, dz: cell, nx, nz } };
      };
      this.meshNear = buildNear();
      this.meshFar = buildFar(this.far);
    }
  };
})();
