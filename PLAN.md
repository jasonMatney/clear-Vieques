# ClearVieques — plan (first deliverable)

A photoreal shallow-water study over **real** Vieques bathymetry. Day scene at Playa Caracas is implemented; Mosquito Bay (night, bioluminescence) is data-ready and is the next milestone.

## 1. Data subset

| | |
|---|---|
| **Elevation source** | NOAA NCEI **CUDEM 1/9 arc-second** (≈3.4 m) topo-bathy, Puerto Rico, 2022v2 (tiles `ncei19_n18x25_w065x50`, plus `…w065x75` for Mosquito's far edge). DOI 10.25921/ds9v-ky35. Cloud-optimised GeoTIFF on the AWS open-data bucket `noaa-nos-coastal-lidar-pds`; read by windowed range requests (≈41 MB for a clean rebuild). |
| **Vertical datum** | NAD83 / PRVD02 orthometric heights, treated as mean sea level (0 m). PRVD02↔local MSL is ≲0.1 m and not applied. |
| **Per site** | `near` grid 4 m spacing, 4 km × 3 km; `far` grid 20 m spacing, 14 km × 10 km. Int16 centimetres, row 0 = north. Blended over 160 m in the shader. |
| **Cross-check** | NOAA NGS 2019 topobathy lidar DEM (validation only): CUDEM is +0.13 m (Caracas) / +0.10 m (Mosquito) higher, RMS 0.20 / 0.18 m; worst on steep reef fronts and forested slopes. |
| **OSM** | Sparse only: 45 tracks/paths + 3 picnic shelters (Caracas), 41 tracks (Mosquito). Tracks are rasterised into the ground material; shelters are not drawn yet. |

**Deviation from the brief, stated plainly:** the "2016 NOAA NGS topobathy lidar DEM for Puerto Rico, Vieques, Culebra" does not cover Vieques (per the data pipeline's search), and no CUDEM 1/3″ tile exists over Vieques. CUDEM 1/9″ is therefore the sole source of every elevation value; the 2019 NGS lidar cross-checks it.

**Sites** (origins chosen from the DEM, not from the approximate coordinates in the brief):

* **Playa Caracas** — origin **18.1120° N, 65.3841° W**, the arc-length midpoint of the 0 m waterline between the headland tips (W tip x −926, E tip x +942 m). The brief's 18.108 N, 65.386 W falls ~490 m offshore. Crescent 1.67 km long, mean beach slope 1:15, a very shallow terrace (0.2–1.5 m deep over the first ~100 m off the north shore near x ≈ 330), bay interior 3–7 m, an islet with a sand spit on the axis.
* **Mosquito Bay** — origin **18.1020° N, 65.4451° W** (lagoon centroid). Lagoon 61.7 ha, mean depth 1.8 m, max 4.0 m; inlet neck 79 m wide at (x 526, z 398), channel 628 m long.

## 2. Coordinate frame

Local azimuthal-equidistant projection centred on each site's origin (`+proj=aeqd +datum=WGS84`), then:

* **x = east (m), y = up (m above MSL), z = SOUTH (m)** — right-handed; **north is −z**. Compass yaw θ looks along `(sin θ, ·, −cos θ)`.
* Grid cell (i, j) centre = `(x0 + (i+½)·dx, z0 + (j+½)·dz)`; textures are sampled bilinearly, so the DEM is continuous everywhere the water shader marches a ray.
* Reverse-Z infinite projection, float32 everywhere; the wave phase is quantised to a 256 s loop so nothing drifts.

## 3. Optics — day vs night

**Day (Playa Caracas, implemented).** All optics are GPU code.

* **Waves:** 3 tiled FFT cascades (250 m / 34 m / 4.7 m, 256² each, rotated 0°/37°/−71° to hide tiling) from a JONSWAP wind sea (fetch-limited; peak and Phillips constant follow the *Wind* and *Wave energy* sliders) plus a narrow swell, finite-depth + capillary dispersion, crest-sharpening (choppy) displacement, Jacobian whitecaps with persistence. GPU field statistics are checked against the analytic spectrum (`CV.app.waves.expected()` vs `readStats()`).
* **Ripples:** 512² wave-equation heightfield, camera-following toroidal window, wave speed `√(g·depth)` from the real DEM (they refract into and die on the beach); click/drag injects impulses.
* **Surface:** exact unpolarised Fresnel; reflected rays are ray-marched against the DEM (headland and forest reflect); GGX sun glitter with variance-loss roughness.
* **Refraction & absorption:** the refracted view ray is ray-marched to the **actual seabed height field**; Beer–Lambert per RGB channel with Jerlov-type coefficients (slider walks I → 9C); water-leaving radiance `(1−F)/n²`; column in-scatter saturates to deep-water colour. Validated: rendered colour vs depth follows the analytic prediction (within ~15 % in the shallows, 2–5 % below 6 m): pale sand → jade → turquoise → deep blue, red gone by ~3 m.
* **Caustics:** photons from a wavy surface are refracted toward the sun and marched to the seabed; per-triangle flux (source area ÷ footprint area) is splatted into a 2048² map (mip 1 is the sampled base) and looked up at the real seafloor position with sun-penumbra blur growing with depth.
* **Sky:** CPU single-scattering atmosphere (Rayleigh + Mie + ozone) → LUT; calibrated to diffuse ≈ 17 % of global irradiance; procedural cumulus feed the reflections. Khronos PBR-Neutral tone mapping.
* **Land:** procedural dry-forest crowns, sea-grape scrub, wind-rippled sand, wet swash band, sun shadows marched on the DEM. CUDEM is bare-earth, so canopy is procedural.

**Night (Mosquito Bay, next).** Same water optics with a moon-lit sky (moon-phase slider scales sky/moon radiance and glow visibility), dark mangroves, and a *Pyrodinium*-style bioluminescence term driven by agitation: the ripple sim already writes an agitation channel; camera motion and a virtual paddle will inject into it, and a short-lived blue-green emission (≈ 480–500 nm) is added to the refracted radiance where agitation is high. Data (`mosquito.*`) is packed.

## 4. Architecture

`index.html` + classic scripts (no bundler, no ES modules, so it also runs from `file://`) + `data/terrain.js` (base64 Int16 grids).

```
sky.js (CPU LUT) → waves.js [evolve → FFT rows/cols → assemble → mips] → ripples.js [step ×2 → out]
                 → caustics.js [photons → splat → mips] → render.js [sky → terrain → water → tone-map]
```

Adaptive resolution is driven by GPU timestamp queries (render scale first, then quality tiers: photon count and MSAA).

## 5. Roadmap

1. Mosquito Bay scene: night sky/moon, mangrove shading, bioluminescence, paddle.
2. Real benthic classes (NOAA benthic habitat maps) instead of procedural sand/seagrass/hardground.
3. Draw OSM shelters; wave refraction/shoaling over the real bathymetry; underwater camera.
