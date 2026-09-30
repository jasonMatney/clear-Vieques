// ClearVieques — analytic atmosphere on the CPU (single scattering: Rayleigh + Mie + ozone).
// Produces: a sky radiance LUT (azimuth-from-sun x elevation), sun colour after extinction, sky irradiance and horizon colour.
// Scene units: a white Lambertian surface facing a sun of irradiance E has radiance E/pi; we use E_TOA = pi, so
// "white surface facing the sun => radiance ~ 1.0 x transmittance".
(function () {
  'use strict';
  const CV = window.CV;

  const R_E = 6371e3, R_A = R_E + 60e3;
  const H_R = 8000, H_M = 1200;
  const BETA_R = [5.802e-6, 13.558e-6, 33.1e-6];
  const BETA_O = [0.650e-6, 1.881e-6, 0.085e-6]; // ozone absorption (peak density)
  const MIE_G = 0.78;

  // Far intersection distance of ray (p, d) with the sphere of radius r centred at the origin (Earth centre).
  function farHit(px, py, dx, dy, r) {
    const b = px * dx + py * dy, c = px * px + py * py - r * r, disc = b * b - c;
    if (disc < 0) return 0;
    return -b + Math.sqrt(disc);
  }

  CV.Sky = class Sky {
    constructor(gpu, W = 64, H = 48) {
      this.gpu = gpu; this.W = W; this.H = H;
      this.tex = gpu.device.createTexture({ label: 'skyLUT', size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.sunTrans = [1, 1, 1]; this.skyIrr = [0.3, 0.4, 0.6]; this.horizon = [0.6, 0.7, 0.8]; this.zenith = [0.1, 0.2, 0.5];
      this.lastKey = '';
      this.haze = 2.4; this.multi = 3.0; // multi ≈ multiple scattering + aerosol + ground bounce (calibrated: diffuse ≈ 17% of global, zenith ≈ 0.22 x sunlit white)
    }

    // Optical depth (per channel) along a ray from (px,py) in direction (dx,dy) up to the top of the atmosphere.
    tauToTop(px, py, dx, dy, n) {
      const len = farHit(px, py, dx, dy, R_A), ds = len / n; let dR = 0, dM = 0, dO = 0;
      for (let i = 0; i < n; i++) {
        const t = (i + 0.5) * ds, x = px + dx * t, y = py + dy * t, h = Math.hypot(x, y) - R_E;
        dR += Math.exp(-h / H_R) * ds; dM += Math.exp(-h / H_M) * ds; dO += Math.max(0, 1 - Math.abs(h - 25000) / 15000) * ds;
      }
      return { dR, dM, dO };
    }

    update(sunElev, haze = this.haze) {
      const key = sunElev.toFixed(4) + '|' + haze.toFixed(3);
      if (key === this.lastKey) return false; this.lastKey = key; this.haze = haze;
      const { W, H } = this, mieS = 3.996e-6 * 1.7 * haze, mieE = mieS / 0.9;
      const sx = Math.cos(sunElev), sy = Math.sin(sunElev); // sun dir in the x'/y' plane
      const oy = R_E + 8; // observer 8 m up
      const NV = 28, NL = 7;
      const lut = new Float32Array(W * H * 4);
      const pR = (c) => 3 / (16 * Math.PI) * (1 + c * c);
      const g = MIE_G, pM = (c) => 3 / (8 * Math.PI) * ((1 - g * g) * (1 + c * c)) / ((2 + g * g) * Math.pow(1 + g * g - 2 * g * c, 1.5));

      let irr = [0, 0, 0], sumHorizon = [0, 0, 0], nHorizon = 0, zen = [0, 0, 0];
      for (let j = 0; j < H; j++) {
        const v = (j + 0.5) / H, e = (Math.PI / 2) * v * v; // non-linear elevation mapping (dense near horizon)
        const ce = Math.cos(e), se = Math.sin(e);
        for (let i = 0; i < W; i++) {
          const phi = ((i + 0.5) / W) * Math.PI;
          const vx = ce * Math.cos(phi), vy = se, vz = ce * Math.sin(phi);
          const cosG = vx * sx + vy * sy; // angle to sun
          // integrate along the view ray in the plane containing the earth centre: use horizontal comp magnitude
          const dxh = Math.hypot(vx, vz); // horizontal magnitude (view ray lies in a plane through earth centre)
          const tMax = farHit(0, oy, dxh, vy, R_A);
          let sumR = [0, 0, 0], sumM = [0, 0, 0], tauR = 0, tauM = 0, tauO = 0;
          let prevT = 0;
          for (let k = 0; k < NV; k++) {
            const u = (k + 0.5) / NV, t = tMax * u * u, ds = tMax * ((k + 1) / NV) ** 2 - tMax * (k / NV) ** 2;
            // sample position in the (horizontal, up) plane
            const px = dxh * t, py = oy + vy * t, h = Math.hypot(px, py) - R_E;
            const dR = Math.exp(-h / H_R) * ds, dM = Math.exp(-h / H_M) * ds, dO = Math.max(0, 1 - Math.abs(h - 25000) / 15000) * ds;
            tauR += dR; tauM += dM; tauO += dO;
            // optical depth toward the sun from the sample point. Sun dir lies in the sun's vertical plane; the sample point sits
            // at horizontal offset px along the view azimuth. Use the local up at the sample (rotate frame): sun elevation relative
            // to local horizontal is nearly unchanged (curvature term):
            const ang = Math.atan2(px, py); // earth-centre angle of the sample
            // local elevation of sun at that point (small correction): treat sun dir fixed in space, local up rotates with 'ang' along view azimuth
            const ux = Math.sin(ang) * (vx / (dxh || 1)), uz = Math.sin(ang) * (vz / (dxh || 1)), uy = Math.cos(ang);
            const sunUp = sx * ux + sy * uy; // cos(zenith angle of sun at sample point)
            const localElev = Math.asin(Math.max(-1, Math.min(1, sunUp)));
            const lt = this.tauToTop(0, R_E + h, Math.cos(localElev), Math.sin(localElev), NL);
            const oR = tauR + lt.dR, oM = tauM + lt.dM, oO = tauO + lt.dO;
            const sh = localElev < -0.02 ? 0 : 1; // sun below local horizon
            const pr = pR(cosG), pm = pM(cosG);
            for (let c = 0; c < 3; c++) {
              const T = Math.exp(-(BETA_R[c] * oR + mieE * oM + BETA_O[c] * oO)) * sh;
              sumR[c] += T * dR * pr; sumM[c] += T * dM * pm;
            }
          }
          const o = (j * W + i) * 4;
          for (let c = 0; c < 3; c++) {
            // E_TOA = pi  ->  multiply by pi. 'multi' fudges multiple scattering.
            const L = Math.PI * (BETA_R[c] * sumR[c] + mieS * sumM[c]) * this.multi;
            lut[o + c] = L;
          }
          lut[o + 3] = 1;
          // accumulate irradiance on a horizontal plane: E = 2 * sum L sin e cos e de dphi   (phi symmetric)
          const de = (Math.PI / 2) * 2 * v / H, dphi = Math.PI / W, wgt = se * ce * de * dphi * 2;
          for (let c = 0; c < 3; c++) irr[c] += lut[o + c] * wgt;
          if (j === 2 && phi > 1.2) { for (let c = 0; c < 3; c++) sumHorizon[c] += lut[o + c]; nHorizon++; }
          if (j === H - 1 && i === 0) zen = [lut[o], lut[o + 1], lut[o + 2]];
        }
      }
      // sun transmittance at the ground
      const st = this.tauToTop(0, oy, sx, sy, 24);
      this.sunTrans = [0, 1, 2].map(c => Math.exp(-(BETA_R[c] * st.dR + mieE * st.dM + BETA_O[c] * st.dO)));
      this.skyIrr = irr; this.horizon = sumHorizon.map(x => x / Math.max(1, nHorizon)); this.zenith = zen;
      this.lut = lut;
      this.gpu.queue.writeTexture({ texture: this.tex }, CV.toHalf(lut), { bytesPerRow: W * 8 }, [W, H]);
      return true;
    }
  };
})();
