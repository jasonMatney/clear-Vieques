// ClearVieques — site configuration. Poses are in the local ENU frame of each site's DEM (x east, y up, z south, metres).
// yaw = compass bearing (deg), pitch = degrees above the horizon.
(function () {
  'use strict';
  const CV = window.CV;
  // Per site: terrain id, light (sun for day; moon az/elev for night), sea state, default water clarity, which material set the land shader uses
  // (matSet 0 = dry-forest coast, 1 = mangrove lagoon) and whether the water carries bioluminescent dinoflagellates (biolum 0..1).
  CV.SITES = {
    caracas: {
      id: 'caracas', matSet: 0, defaultMode: 'day', biolum: 0,
      name: 'Playa Caracas', sub: 'Red Beach · south coast of Vieques',
      // hero: low over the water, looking toward the headland, sun behind the camera (tuned against the real DEM)
      hero: { pos: [330, 1.8, 55], yaw: 62, pitch: -4.5, fov: 60 },  // 1.9 m of water ~110 m off the north shore, looking ENE along the crescent to the forested headland hill
      heroNight: { pos: [330, 1.8, 55], yaw: 236, pitch: 9, fov: 60 },
      sunAz: 242, sunElev: 58,                                           // sun behind the camera (WSW)
      moonAz: 232, moonElev: 30,
      windFrom: 70,    // trade-wind sea comes from ENE
      swellFrom: 165,  // gentle SSE swell (refined from the shore normal once the DEM is analysed)
      wind: 3.6, energy: 0.35, turbidity: 0.16, swell: 1, fetchBase: 1500, fetchGain: 9000,
    },
    mosquito: {
      id: 'mosquito', matSet: 1, defaultMode: 'night', biolum: 1,
      name: 'Mosquito Bay', sub: 'Puerto Mosquito · bioluminescent lagoon',
      lagoon: { seed: [1.7, -0.5], box: [-660, -330, 590, 410] },       // flood-fill seed and bounding box (x0, z0, x1, z1) of the enclosed lagoon, cut at the inlet neck
      // day: the lagoon looking NE toward Punta Ferro with the sun behind; night: NW across the water with the moon in frame
      hero: { pos: [-20, 1.6, 120], yaw: 62, pitch: 4, fov: 62 },
      heroNight: { pos: [-20, 1.6, 120], yaw: 315, pitch: 8, fov: 62 },
      sunAz: 240, sunElev: 52, moonAz: 305, moonElev: 26,
      windFrom: 70, swellFrom: 160,
      wind: 2.4, energy: 0.10, turbidity: 0.62, swell: 0, fetchBase: 350, fetchGain: 1500,   // enclosed lagoon: short fetch, no ocean swell
    },
  };
})();
