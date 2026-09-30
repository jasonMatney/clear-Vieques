# ClearVieques — plan (first deliverable)

A photoreal shallow-water study over **real** Vieques bathymetry. Both scenes are implemented: Playa Caracas by day and Mosquito Bay by night (moon, stars, mangroves, bioluminescence).

## 1. Data subset

| | |
|---|---|
| **Elevation source** | NOAA NCEI **CUDEM 1/9 arc-second** (≈3.4 m) topo-bathy, Puerto Rico, 2022v2 (tiles `ncei19_n18x25_w065x50`, plus `…w065x75` for Mosquito's far edge). DOI 10.25921/ds9v-ky35. Cloud-optimised GeoTIFF on the AWS open-data bucket `noaa-nos-coastal-lidar-pds`; read by windowed range requests (≈41 MB for a clean rebuild). |
| **Vertical datum** | NAD83 / PRVD02 orthometric heights, treated as mean sea level (0 m). PRVD02↔local MSL is ≲0.1 m and not applied. |
| **Per site** | `near` grid 4 m spacing, 4 km × 3 km; `far` grid 20 m spacing, 14 km × 10 km. Int16 centimetres, row 0 = north. Blended over 160 m in the shader. |
| **Cross-check** | NOAA NGS 2019 topobathy lidar DEM (validation only): CUDEM is +0.13 m (Caracas) / +0.10 m (Mosquito) higher, RMS 0.20 / 0.18 m; worst on steep reef fronts and forested slopes. |
| **OSM** | Sparse only: 45 tracks/paths + 3 picnic shelters (Caracas), 41 tracks (Mosquito). Tracks are rasterised into the ground material; the shelters are drawn as small instanced meshes on their real footprints. |

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

**Night (Mosquito Bay, implemented).** The moon is the "sun" of the night scene: the same shader paths (direct light, sky LUT, glitter, shadows, clouds) run with a light whose strength follows the lunar phase law.

* **Moon and sky:** relative illuminance from Allen's phase law (quarter ≈ 9 % of full, crescent ≈ 2 %); a photographic gain stands in for dark adaptation and compensates only ~55 % of the change, so a thin moon is genuinely darker. The scattered-light sky is dimmed relative to direct moonlight so the night sky stays deep blue with stars. The disc is a lit sphere with the true phase angle (waxing right, waning left), maria and earthshine; procedural stars (one hashed star per cell, three layers, ~1.25 px) and a Milky Way band with lanes. The grade desaturates and shifts toward blue (Purkinje) below ~2 in luminance.
* **Mangroves:** a flood-fill of the DEM's enclosed lagoon gives a distance-to-lagoon field (stored in the aux texture); land that is low, gentle and within ~100–200 m of it is mangrove — a closed, dark, glossy canopy from the same crown bake as the dry forest, allowed to stand in the intertidal shallows. The lagoon floor is mud and seagrass.
* **Bioluminescence:** the ripple sim's agitation channel (0..1) drives emission: `L = colour(0.035, 0.70, 0.95) x gain x [0.95 A^1.2 (patchy plankton density) (ripple energy) (1 - exp(-depth/0.45)) + 3.4 sparks]`, times (1 - F)/n^2 and Beer–Lambert. Sparks are ~7.5 cm cells that each flash at a random phase (~0.15 s), with a probability that grows with agitation and fade out beyond ~40 m. Sources: clicked ripples, a **paddle** (pointer strokes, interpolated so fast drags stay continuous), the **camera as a boat** (bow wave and V ahead of it), and breaking waves / shore swash. The emitted light is fixed, but exposure adapts, so the glow is scaled by (FULL/K)^0.82 (capped): the moon-phase slider changes how visible it is.
* **Site defaults:** the lagoon is enclosed — short fetch (350 m + 1500 m x energy) and no ocean swell — so it is calm, with mirror-like moon reflections.

## 4. Architecture

`index.html` + classic scripts (no bundler, no ES modules, so it also runs from `file://`) + `data/terrain.js` (base64 Int16 grids).

```
per site (load / site switch):  terrain.js (DEM, aux, lagoon field) -> canopy.js [bake crowns 2 m -> stand mean 4 m]
per frame:  sky.js (CPU LUT) -> waves.js [evolve -> FFT rows/cols -> assemble -> mips] -> ripples.js [step x2 -> out, skipped when settled]
            -> caustics.js [photons -> splat (1024 or 2048) -> mips] -> render.js [sky -> terrain tiles -> shelters -> water]
            -> post [grade to LDR -> edge-adaptive upscale]  (or grade straight to the canvas at native resolution)
```

Adaptive resolution walks a ladder of `(quality tier, render scale)` rungs from GPU timestamp queries (with hysteresis and a memory of rungs that failed); tiers set the caustic photon count / map size / refresh interval and MSAA. Site switching (`switchSite`) decodes the new terrain, re-bakes its canopy, rebinds the scene and ripple bind groups and destroys the old textures (~0.8 s).

## 5. Roadmap

1. Real benthic classes (NOAA benthic habitat maps) instead of procedural sand/seagrass/hard-ground; mangrove prop roots in the shallows.
2. Wave refraction/shoaling over the real bathymetry; an underwater camera.
3. Moon and sun from an ephemeris (date, time, latitude) instead of sliders; tides from the vertical-datum offsets.
4. Cross-browser / GPU validation (Safari, Firefox, integrated GPUs) and the half-float DEM fallback path.
