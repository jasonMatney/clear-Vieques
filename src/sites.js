// ClearVieques — site configuration. Poses are in the local ENU frame of each site's DEM (x east, y up, z south, metres).
// yaw = compass bearing (deg), pitch = degrees above the horizon.
(function () {
  'use strict';
  const CV = window.CV;
  CV.SITES = {
    caracas: {
      id: 'caracas', available: true, mode: 'day',
      name: 'Playa Caracas', sub: 'Red Beach · south coast of Vieques',
      // hero: low over the water, looking toward the headland, sun behind the camera (tuned against the real DEM)
      hero: { pos: [330, 1.8, 55], yaw: 62, pitch: -4.5, fov: 60 },  // 1.9 m of water ~110 m off the north shore, looking ENE along the crescent to the forested headland hill
      sunAz: 242, sunElev: 58,                                           // sun behind the camera (WSW)
      windFrom: 70,    // trade-wind sea comes from ENE
      swellFrom: 165,  // gentle SSE swell (refined from the shore normal once the DEM is analysed)
    },
    mosquito: {
      id: 'mosquito', available: false, mode: 'night',
      name: 'Mosquito Bay', sub: 'Puerto Mosquito · bioluminescent lagoon',
      hero: { pos: [0, 1.5, 0], yaw: 0, pitch: -5, fov: 62 }, sunAz: 200, sunElev: 30, windFrom: 70, swellFrom: 160,
    },
  };
})();
