// ClearVieques — night lighting model. The moon is the "sun" of the night scene: the same shader paths (direct light, sky LUT, glitter, shadows)
// run with a light whose strength follows the lunar phase law, and a photographic gain stands in for dark adaptation.
(function () {
  'use strict';
  const CV = window.CV;

  const Night = CV.Night = {
    // Slider phase p in [0,1]: 0 new, 0.25 first quarter, 0.5 full, 0.75 last quarter, 1 new. Phase angle 0 (full) .. pi (new).
    phaseAngle(p) { return Math.abs(p - 0.5) * 2 * Math.PI; },
    // Relative illuminance of the moon (full = 1) by Allen's apparent-magnitude law: m = m_full + 0.026|a| + 4e-9 a^4 (a in degrees), so a
    // quarter moon gives ~9 % of a full moon and a thin crescent ~2 %, and the light dies away toward new.
    moonRel(p) {
      const a = Night.phaseAngle(p) * 180 / Math.PI;
      const m = 0.026 * a + 4e-9 * Math.pow(a, 4);
      return Math.pow(10, -0.4 * m) * (1 - CV.smoothstep(150, 178, a));
    },
    // Scale applied to every light source of the scene (sun -> moon). Eyes and cameras adapt: only ~55 % of the change in log illuminance is
    // compensated, so a thin moon is genuinely darker (and glow more visible) than a full one. FLOOR = starlight / airglow.
    FULL: 0.22, FLOOR: 0.0022,
    gain(p) { return Night.FULL * Math.pow(Night.moonRel(p), 0.55) + Night.FLOOR; },
    // 1 = dark sky full of stars; a full moon washes out the faint ones but the bright stars always remain
    starVis(p) { return 0.30 + 0.70 * (1 - CV.smoothstep(0.05, 0.85, Night.moonRel(p))); },
    // The scattered-light sky (and its haze/halo) is dimmed relative to the direct moonlight on the ground: to the dark-adapted eye the night sky
    // stays deep blue and near-black around the moon, whereas the same physical ratio at day-for-night exposure would read as dusk.
    SKY: 0.32,
    // Signed phase angle for the disc shader: > 0 waxing (lit limb on the right in the northern hemisphere), < 0 waning.
    signedAlpha(p) { const a = Night.phaseAngle(p); return p < 0.5 ? a : -a; },
  };
})();
