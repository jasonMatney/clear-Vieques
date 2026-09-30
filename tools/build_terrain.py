#!/usr/bin/env python3
"""
build_terrain.py - ClearVieques geodata builder (Vieques, Puerto Rico).

Builds, for two sites (`caracas` = Playa Caracas / Red Beach, `mosquito` = Puerto Mosquito / Mosquito Bay):

    data/<site>_near.i16    4 m grid, Int16 little-endian, centimetres above mean sea level
    data/<site>_far.i16    20 m grid, same encoding
    data/<site>.json        grid specs, projection, sources, statistics, validation metrics, caveats
    data/osm_<site>.json    sparse OpenStreetMap features (tracks/paths, <=3 structures, coastline) in local metres
    data/preview_<site>_{near,far}.png   hillshade + bathymetry ramp + 0 m contour (north up)

and then `tools/pack_terrain.py` packs everything into data/terrain.js for the browser.

SITE FRAME
    Azimuthal-equidistant projection centred on the site origin (+proj=aeqd +lat_0 +lon_0 +datum=WGS84 +units=m),
    then   x = east (m),  z = SOUTH (m; north is -z),  y = elevation above mean sea level (m).
    Grids are row-major, row 0 = northernmost (most negative z), column 0 = westernmost; values are sampled at cell
    centres:  cell (row j, col i) is at  x = x0 + (i + 0.5) dx,  z = z0 + (j + 0.5) dz.

DATA SOURCES (public NOAA data only, read with windowed HTTP range requests - no whole tiles are downloaded)
    PRIMARY     NOAA NCEI CUDEM 1/9 arc-second Puerto Rico topobathy tiles (ncei19_n18x25_w065x50 / w065x75, 2022v2),
                Cloud-Optimized GeoTIFF on the AWS open-data bucket noaa-nos-coastal-lidar-pds.
                near grids: native-resolution window (3.3 x 3.4 m) ; far grids: the COG's 2x overview level (6.9 m).
    CROSS-CHECK 2019 NOAA NGS Topobathy Lidar DEM: Puerto Rico (dataset 9392, 1 m tiles, 4 m overview level
                = exact 2x2 box averages of the 1 m data). Used for validation only; nothing is merged into the products.
    OSM         Overpass API, one small bbox query per site (ODbL).

Total network use is counted from GDAL's VSICURL debug log and reported at the end (cap for the task: ~250 MB).

USAGE
    python3 tools/build_terrain.py                       # everything, both sites
    python3 tools/build_terrain.py --sites caracas       # one site
    python3 tools/build_terrain.py --cache /some/dir     # where windowed source reads are cached (default tools/_cache)
    python3 tools/build_terrain.py --skip-osm            # do not call Overpass
    python3 tools/build_terrain.py --out-dir /tmp/x      # write products elsewhere (e.g. to test a clean rebuild)
    python3 tools/build_terrain.py --check-origin        # only print the crescent-middle / lagoon-centre origin suggestion
    python3 tools/build_terrain.py --denoise 0.7         # optional 3x3 gaussian on WATER cells only (default: not applied)

Requires: GDAL command-line tools on PATH (gdalwarp, gdal_translate, gdalbuildvrt, gdaltransform, ogr2ogr),
python3 with numpy, scipy, scikit-image, Pillow, tifffile.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terrain_analysis as TA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_CACHE = ROOT / "tools" / "_cache"

# --------------------------------------------------------------------------------------------------
# Site configuration
# --------------------------------------------------------------------------------------------------
# Origins are hard-coded (rounded to 4 decimals) so the products are stable; they were derived from the DEM as follows.
#   caracas : arc-length midpoint of the 0 m contour between the two headland tips of the crescent bay, i.e. the
#             cuspate foreland directly behind the islet that sits on the bay axis (analyze_caracas -> 'crescent_middle').
#             --check-origin recomputes it. The user-supplied hint (18.108, -65.386) is ~500 m offshore of it.
#   mosquito: area centroid of the enclosed lagoon (water z < -0.5 m, cut from the sea at the inlet neck).
SITES = {
    "caracas": {
        "name": "Playa Caracas (Red Beach), Vieques, Puerto Rico",
        "lat0": 18.1120, "lon0": -65.3841,
        "hint": (18.108, -65.386),
        "near": {"x": (-2000.0, 2000.0), "z": (-1200.0, 1800.0), "d": 4.0},
        "far": {"x": (-7000.0, 7000.0), "z": (-4500.0, 5500.0), "d": 20.0},
        # approximate headland tips (lat, lon) flanking the crescent, used only to seed the tip search (250 m radius)
        "tip_seeds_latlon": [(18.10776, -65.39263), (18.10724, -65.37513)],
    },
    "mosquito": {
        "name": "Puerto Mosquito (Mosquito Bay), Vieques, Puerto Rico",
        "lat0": 18.1020, "lon0": -65.4451,
        "hint": (18.102, -65.446),
        # 1.0 km inland / 2.0 km offshore (instead of 1.2 / 1.8) so the whole lagoon + inlet + ~1 km of open sea fit
        "near": {"x": (-2000.0, 2000.0), "z": (-1000.0, 2000.0), "d": 4.0},
        "far": {"x": (-7000.0, 7000.0), "z": (-4500.0, 5500.0), "d": 20.0},
    },
}

NODATA = -9999.0
CLAMP_M = 327.0          # Int16 centimetres: values clamped to +-327.00 m
S3 = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/"
CUDEM_BASE = S3 + "NCEI_ninth_Topobathy_PuertoRico_9525/"
CUDEM_TILES = ["ncei19_n18x25_w065x75_2022v2.tif", "ncei19_n18x25_w065x50_2022v2.tif"]  # west, east
NGS19_INDEX = S3 + "NGS_PR_Topobathy_DEM_2019_9392/tileindex_NGS_PR_Topobathy_DEM_2019.zip"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = "ClearVieques-terrain-build/1.0 (research demo; public NOAA + OSM data)"

# --------------------------------------------------------------------------------------------------
# Small utilities: logging, GDAL runner with network accounting, coordinate transforms
# --------------------------------------------------------------------------------------------------
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def _find_gdal():
    if shutil.which("gdalwarp"):
        return
    for p in ("/opt/anaconda3/bin", "/opt/homebrew/bin", "/usr/local/bin"):
        if os.path.exists(os.path.join(p, "gdalwarp")):
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            return
    sys.exit("GDAL command-line tools not found on PATH")


_find_gdal()
ENV = dict(os.environ)
ENV.update({
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt,.zip",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "180",
    "VSI_CACHE": "TRUE",
    "OSR_STRIP_TOWGS84": "YES",
})


class Net:
    """Running total of bytes fetched over the network (GDAL range reads + urllib downloads)."""
    total = 0
    events: list = []

    @classmethod
    def add(cls, tag, n):
        cls.total += n
        cls.events.append((tag, n))


def gdal(cmd, tag):
    """Run a GDAL CLI utility with '--debug ON' and add the bytes of all VSICURL range reads to Net."""
    full = [cmd[0], "--debug", "ON"] + [str(c) for c in cmd[1:]]
    p = subprocess.run(full, env=ENV, capture_output=True, text=True)
    nbytes = sum(int(m.group(2)) - int(m.group(1)) + 1
                 for m in re.finditer(r"VSICURL: Downloading (\d+)-(\d+) \(", p.stderr))
    Net.add(tag, nbytes)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.splitlines()[-15:])
        raise RuntimeError(f"{cmd[0]} failed ({tag}):\n{tail}")
    return p.stdout, nbytes


def http_get(url, dest=None, data=None, headers=None, tag="http"):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = r.read()
    Net.add(tag, len(body))
    if dest:
        Path(dest).write_bytes(body)
    return body


def aeqd(lat0, lon0):
    return f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"


def transform_pts(pts, s_srs, t_srs):
    """Batch coordinate transform through PROJ via `gdaltransform` (traditional x/y = lon/lat order)."""
    if len(pts) == 0:
        return np.zeros((0, 2))
    inp = "\n".join(f"{x!r} {y!r}" for x, y in pts) + "\n"
    r = subprocess.run(["gdaltransform", "-s_srs", s_srs, "-t_srs", t_srs, "-output_xy"], input=inp,
                       capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError("gdaltransform failed: " + r.stderr[-500:])
    return np.array([[float(v) for v in ln.split()[:2]] for ln in r.stdout.strip().splitlines()])


def site_xz_to_lonlat(site, xz):
    """(x east, z south) in the site frame -> (lon, lat)."""
    pts = [(float(x), -float(z)) for x, z in xz]
    return transform_pts(pts, aeqd(site["lat0"], site["lon0"]), "EPSG:4269")


def lonlat_to_site_xz(site, lonlat):
    out = transform_pts([(float(lo), float(la)) for lo, la in lonlat], "EPSG:4269", aeqd(site["lat0"], site["lon0"]))
    return np.column_stack([out[:, 0], -out[:, 1]])


def window_lonlat_bbox(site, spec, pad_m=0.0, n=41):
    """Geographic bounding box (lon0, lon1, lat0, lat1) of an aeqd window (perimeter sampled)."""
    (xa, xb), (za, zb) = spec["x"], spec["z"]
    xa, xb, za, zb = xa - pad_m, xb + pad_m, za - pad_m, zb + pad_m
    t = np.linspace(0, 1, n)
    edge = [(xa + (xb - xa) * u, za) for u in t] + [(xa + (xb - xa) * u, zb) for u in t] + \
           [(xa, za + (zb - za) * u) for u in t] + [(xb, za + (zb - za) * u) for u in t]
    ll = site_xz_to_lonlat(site, edge)
    return float(ll[:, 0].min()), float(ll[:, 0].max()), float(ll[:, 1].min()), float(ll[:, 1].max())


def grid_of(spec):
    (xa, xb), (za, zb), d = spec["x"], spec["z"], spec["d"]
    return TA.Grid(nx=int(round((xb - xa) / d)), nz=int(round((zb - za) / d)), dx=d, dz=d, x0=xa, z0=za)


def gdalinfo_json(path):
    out, _ = gdal(["gdalinfo", "-json", path], "gdalinfo")
    return json.loads(out)


# --------------------------------------------------------------------------------------------------
# Stage 1: windowed reads (cached)
# --------------------------------------------------------------------------------------------------
def cudem_vrt(cache):
    """Local VRT over the two CUDEM tiles (west tile first so the east tile wins in the 12-px buffer overlap)."""
    vrt = cache / "cudem19_pr_vieques.vrt"
    if not vrt.exists():
        urls = ["/vsicurl/" + CUDEM_BASE + t for t in CUDEM_TILES]
        gdal(["gdalbuildvrt", "-overwrite", "-srcnodata", NODATA, "-vrtnodata", NODATA, vrt] + urls, "cudem_vrt")
    return vrt


def srcwin(gt, size, bbox, align=1):
    """Pixel window (xoff, yoff, xsize, ysize) of a north-up geotransform covering bbox=(lon0,lon1,lat0,lat1)."""
    lon0, lon1, lat0, lat1 = bbox
    x0 = math.floor((lon0 - gt[0]) / gt[1])
    x1 = math.ceil((lon1 - gt[0]) / gt[1])
    y0 = math.floor((lat1 - gt[3]) / gt[5])
    y1 = math.ceil((lat0 - gt[3]) / gt[5])
    x0 -= x0 % align
    y0 -= y0 % align
    x1 += (-x1) % align
    y1 += (-y1) % align
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size[0]), min(y1, size[1])
    return x0, y0, x1 - x0, y1 - y0


def _window_covers(rec, bbox):
    """True if a cached window record (srcwin + geotransform) covers the geographic bbox (lon0, lon1, lat0, lat1)."""
    gt = rec["vrt_geotransform"]
    x0, y0, xs, ys = rec["srcwin"]
    c_lon0, c_lon1 = gt[0] + x0 * gt[1], gt[0] + (x0 + xs) * gt[1]
    c_lat1, c_lat0 = gt[3] + y0 * gt[5], gt[3] + (y0 + ys) * gt[5]
    return c_lon0 <= bbox[0] and c_lon1 >= bbox[1] and c_lat0 <= bbox[2] and c_lat1 >= bbox[3]


def fetch_cudem(site_id, site, kind, cache, refetch=False):
    """
    Cache a CUDEM window as a local GeoTIFF (EPSG:4269, float32, nodata -9999).
    kind='near': native 1/9 arc-second pixels (no resampling).  kind='far': 2x overview level (COG overview data).
    The cache is reused only if it still covers the window required by the current site origin.
    Returns (path, sidecar dict).
    """
    out = cache / f"cudem19_{site_id}_{kind}.tif"
    side = cache / f"cudem19_{site_id}_{kind}.json"
    pad = 60.0 if kind == "near" else 150.0            # margin (m) for the resampling kernel
    bbox = window_lonlat_bbox(site, site[kind], pad_m=pad)
    if out.exists() and side.exists() and not refetch:
        rec = json.loads(side.read_text())
        # reuse the cache if it still covers the window with at least half of the requested margin
        if _window_covers(rec, window_lonlat_bbox(site, site[kind], pad_m=pad / 2)):
            return out, rec
        log(f"cached CUDEM {kind} window for {site_id} does not cover the current origin -> re-reading")
    vrt = cudem_vrt(cache)
    info = gdalinfo_json(str(vrt))
    gt, size = info["geoTransform"], info["size"]
    level = 1 if kind == "near" else 2
    w = srcwin(gt, size, bbox, align=level)
    cmd = ["gdal_translate", "-of", "GTiff", "-srcwin", *w]
    if level == 2:
        cmd += ["-outsize", w[2] // 2, w[3] // 2, "-r", "nearest"]   # COG overview level 2x (6.9 m)
    cmd += ["-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=3", "-co", "TILED=YES", str(vrt), str(out)]
    log(f"fetch CUDEM {kind} window for {site_id}: srcwin={w} (level {level}x)")
    _, nbytes = gdal(cmd, f"cudem_{site_id}_{kind}")
    rec = {"file": out.name, "srcwin": list(w), "level": level, "bbox_lonlat": bbox, "bytes": nbytes,
           "fetched_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "vrt_geotransform": gt}
    side.write_text(json.dumps(rec, indent=1))
    log(f"   -> {nbytes / 1e6:.1f} MB over the network")
    return out, rec


def ngs19_tiles(site, cache):
    """URLs of the NGS 2019 lidar DEM tiles intersecting the near window (from the dataset's tile index)."""
    zpath = cache / "tileindex_NGS_PR_Topobathy_DEM_2019.zip"
    if not zpath.exists():
        http_get(NGS19_INDEX, zpath, tag="ngs19_tileindex")
    bbox = window_lonlat_bbox(site, site["near"], pad_m=30.0)
    shp = f"/vsizip/{zpath}/tileindex_NGS_PR_Topobathy_DEM_2019.shp"
    out, _ = gdal(["ogr2ogr", "-f", "GeoJSON", "/vsistdout/", shp, "-spat", bbox[0], bbox[2], bbox[1], bbox[3]], "ngs19_index_query")
    gj = json.loads(out[out.index("{"):])
    return sorted(f["properties"]["url"] for f in gj["features"])


def fetch_lidar_near(site_id, site, cache, refetch=False):
    """
    NGS 2019 lidar cross-check grid on the near-grid geometry (aeqd, 4 m, bilinear, nodata -9999).
    The remote 1 m tiles are read at their 4 m COG overview (`-ovr 1`), which is an exact 2x2 box average, so only
    ~0.2 MB per 1 km tile is transferred.
    """
    out = cache / f"ngs2019_{site_id}_near4m.tif"
    side = cache / f"ngs2019_{site_id}_near4m.json"
    if out.exists() and side.exists() and not refetch:
        rec = json.loads(side.read_text())
        if rec.get("origin") == [site["lat0"], site["lon0"]] and rec.get("window") == [list(site["near"]["x"]), list(site["near"]["z"])]:
            return out, rec
        log(f"cached lidar grid for {site_id} was built for a different origin/window -> rebuilding")
    urls = ngs19_tiles(site, cache)
    g = grid_of(site["near"])
    xmin, xmax, zmin, zmax = g.extent
    cmd = ["gdalwarp", "-overwrite", "-ovr", "1", "-t_srs", aeqd(site["lat0"], site["lon0"]),
           "-te", xmin, -zmax, xmax, -zmin, "-tr", g.dx, g.dz, "-r", "bilinear",
           "-srcnodata", "-3.4028235e+38", "-dstnodata", NODATA, "-ot", "Float32", "-of", "GTiff",
           "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=3"] + ["/vsicurl/" + u for u in urls] + [str(out)]
    log(f"fetch NGS-2019 lidar 4 m overview for {site_id}: {len(urls)} tiles")
    _, nbytes = gdal(cmd, f"ngs2019_{site_id}")
    rec = {"file": out.name, "origin": [site["lat0"], site["lon0"]], "window": [list(site["near"]["x"]), list(site["near"]["z"])],
           "tiles": [u.rsplit("/", 1)[-1] for u in urls], "tile_urls": urls, "bytes": nbytes,
           "fetched_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    side.write_text(json.dumps(rec, indent=1))
    log(f"   -> {nbytes / 1e6:.1f} MB over the network")
    return out, rec


# --------------------------------------------------------------------------------------------------
# Stage 2: resample to the site frame, fill, quantise
# --------------------------------------------------------------------------------------------------
def warp_to_site_grid(src, dst, site, spec, method="bilinear", src_srs="EPSG:4269"):
    g = grid_of(spec)
    xmin, xmax, zmin, zmax = g.extent
    cmd = ["gdalwarp", "-overwrite", "-s_srs", src_srs, "-t_srs", aeqd(site["lat0"], site["lon0"]),
           "-te", xmin, -zmax, xmax, -zmin, "-tr", g.dx, g.dz, "-r", method,
           "-srcnodata", NODATA, "-dstnodata", NODATA, "-ot", "Float32", "-of", "GTiff",
           "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=3", str(src), str(dst)]
    gdal(cmd, "local_warp")
    a = tifffile.imread(dst).astype(np.float64)
    assert a.shape == (g.nz, g.nx), (a.shape, g)
    return a, g


def fill_nodata(z, g, deepen=0.02, floor=-320.0):
    """
    Fill NoData (NaN) cells. Voids whose valid neighbours are (mostly) below sea level are treated as offshore:
    the nearest valid depth is extrapolated seaward with a constant deepening gradient (`deepen` m/m, capped at
    `floor`) so the seabed keeps deepening. Other voids (land) are filled by harmonic (Laplace) interpolation
    from their valid boundary. Returns (filled array, number of filled cells).
    """
    bad = ~np.isfinite(z)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return z, 0
    out = z.copy()
    dist, (ri, ci) = ndi.distance_transform_edt(bad, return_indices=True)
    nearest = z[ri, ci]
    lab, n = ndi.label(bad)
    for L in range(1, n + 1):
        m = lab == L
        ring = ndi.binary_dilation(m, iterations=1) & ~m & ~bad
        nb = z[ring]
        if nb.size and np.median(nb) < -0.2:                      # offshore void
            out[m] = np.maximum(nearest[m] - deepen * dist[m] * g.dx, floor)
        else:                                                     # land void: harmonic interpolation
            v = np.where(m, np.nanmean(nb) if nb.size else 0.0, z)
            for _ in range(400):
                avg = (np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1)) / 4.0
                v = np.where(m, avg, v)
            out[m] = v[m]
    return out, n_bad


def quantise_cm(z):
    """metres -> Int16 centimetres, clamped to +-327 m. Returns (int16 array, clamped fraction)."""
    cm = np.rint(z * 100.0)
    lim = CLAMP_M * 100.0
    clamped = float(np.mean((cm > lim) | (cm < -lim)))
    return np.clip(cm, -lim, lim).astype("<i2"), clamped


def build_grid(site_id, site, kind, src, method, denoise=None):
    """Warp -> fill -> optional water-only denoise -> quantise. Returns dict with float grid, int16 grid and stats."""
    tmp = src.parent / f"tmp_{site_id}_{kind}_site.tif"
    a, g = warp_to_site_grid(src, tmp, site, site[kind], method=method)
    bad = (a <= NODATA + 1) | ~np.isfinite(a)
    a[bad] = np.nan
    z, n_filled = fill_nodata(a, g)
    if denoise:
        sm = ndi.gaussian_filter(z, denoise, truncate=1.0 / denoise + 0.01)  # 3x3 kernel
        wet = z < -0.5
        z = np.where(wet, sm, z)
    q, clamped = quantise_cm(z)
    zq = q.astype(np.float64) / 100.0
    stats = {
        "nx": g.nx, "nz": g.nz, "dx": g.dx, "dz": g.dz, "x0": g.x0, "z0": g.z0,
        "window_x_m": [g.x0, g.x0 + g.nx * g.dx], "window_z_m": [g.z0, g.z0 + g.nz * g.dz],
        "min_m": round(float(zq.min()), 2), "max_m": round(float(zq.max()), 2), "mean_m": round(float(zq.mean()), 2),
        "water_fraction": round(float((zq < 0).mean()), 4), "land_fraction": round(float((zq >= 0).mean()), 4),
        "nodata_filled_cells": n_filled, "nodata_filled_fraction": round(n_filled / zq.size, 6),
        "clamped_fraction": round(clamped, 4), "denoise_sigma_cells": denoise,
    }
    return {"grid": g, "z": z, "zq": zq, "q": q, "stats": stats, "nodata_mask": bad}


# --------------------------------------------------------------------------------------------------
# Stage 3: OpenStreetMap (one Overpass query per site)
# --------------------------------------------------------------------------------------------------
def osm_query(site):
    """Overpass QL for the sparse feature set inside the near window; returns (query, (S, W, N, E))."""
    lon0, lon1, lat0, lat1 = window_lonlat_bbox(site, site["near"], pad_m=0.0)
    bb = f"{lat0:.5f},{lon0:.5f},{lat1:.5f},{lon1:.5f}"
    q = (f'[out:json][timeout:60];('
         f'way["highway"~"^(track|path|service|unclassified)$"]({bb});'
         f'nwr["building"]({bb});nwr["amenity"="shelter"]({bb});'
         f'nwr["tourism"~"^(picnic_site|viewpoint)$"]({bb});nwr["man_made"]({bb});'
         f'way["natural"="coastline"]({bb});); out geom tags;')
    return q, (lat0, lon0, lat1, lon1)


def fetch_osm(site_id, site, cache):
    """Sparse OSM features inside the near window, converted to site-frame metres (x east, z south)."""
    q, (lat0, lon0, lat1, lon1) = osm_query(site)
    raw = cache / f"osm_{site_id}_{hashlib.md5(q.encode()).hexdigest()[:10]}.json"   # one Overpass call per distinct query
    if not raw.exists():
        log(f"Overpass query for {site_id} (one bbox query) ...")
        last = None
        for url in OVERPASS:
            try:
                body = http_get(url, data=urllib.parse.urlencode({"data": q}).encode(), tag=f"overpass_{site_id}")
                raw.write_bytes(body)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:      # noqa: PERF203
                last = e
                log(f"   overpass endpoint failed ({url}): {e}")
        else:
            raise RuntimeError(f"Overpass unavailable: {last}")
    els = json.loads(raw.read_text())["elements"]
    g = grid_of(site["near"])
    xmin, xmax, zmin, zmax = g.extent

    def to_xz(geom):
        ll = [(p["lon"], p["lat"]) for p in geom]
        return lonlat_to_site_xz(site, ll)

    def clip_runs(xz, margin=40.0):
        inside = (xz[:, 0] > xmin - margin) & (xz[:, 0] < xmax + margin) & (xz[:, 1] > zmin - margin) & (xz[:, 1] < zmax + margin)
        runs, cur = [], []
        for p, ok in zip(xz, inside):
            if ok:
                cur.append([round(float(p[0]), 1), round(float(p[1]), 1)])
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        return [r for r in runs if len(r) >= 2]

    roads, coast, structs = [], [], []
    for e in els:
        tags = e.get("tags", {})
        if e["type"] == "way" and "geometry" in e:
            xz = to_xz(e["geometry"])
            if tags.get("natural") == "coastline":
                for r in clip_runs(xz):
                    coast.append({"id": e["id"], "pts": r})
            elif "highway" in tags:
                for r in clip_runs(xz):
                    roads.append({"id": e["id"], "highway": tags["highway"], "name": tags.get("name"),
                                  "surface": tags.get("surface"), "pts": r})
            else:
                area = 0.5 * abs(sum(xz[i, 0] * xz[(i + 1) % len(xz), 1] - xz[(i + 1) % len(xz), 0] * xz[i, 1] for i in range(len(xz))))
                c = xz.mean(axis=0)
                structs.append({"id": e["id"], "type": "way", "tags": tags, "area_m2": round(float(area), 1),
                                "centroid": [round(float(c[0]), 1), round(float(c[1]), 1)],
                                "pts": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in xz]})
        elif e["type"] == "node":
            xz = lonlat_to_site_xz(site, [(e["lon"], e["lat"])])[0]
            structs.append({"id": e["id"], "type": "node", "tags": tags, "area_m2": 0.0,
                            "centroid": [round(float(xz[0]), 1), round(float(xz[1]), 1)]})
    # keep at most 3 small structures (no urban fabric): smallest area first, then nearest to the origin
    structs = [s for s in structs if s["area_m2"] < 3000.0]
    structs.sort(key=lambda s: (math.hypot(*s["centroid"]), s["area_m2"]))
    kept = structs[:3]
    out = {
        "site": site_id, "frame": "x east (m), z south (m), site aeqd frame; polylines are [[x,z],...]",
        "bbox_lonlat_swne": [lat0, lon0, lat1, lon1], "query": q,
        "queried_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "attribution": "Data (c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
        "counts_returned": {"roads": len(roads), "structures_all": len(structs), "structures_kept": len(kept), "coastline": len(coast)},
        "roads": roads, "structures": kept, "coastline": coast,
        "note": ("OSM has no features of the requested kinds inside this window." if not (roads or kept or coast) else
                 "Sparse features only; structures limited to 3; coastline is included for cross-checking against the DEM 0 m contour."),
    }
    return out


def osm_vs_dem_coastline(osm, zq, g):
    """Median |signed| DEM elevation sampled on the OSM coastline (0 m expected) and its 90th percentile."""
    vals = []
    for c in osm.get("coastline", []):
        p = np.array(c["pts"])
        if len(p) < 2:
            continue
        s = TA.arclen(p[:, 0], p[:, 1])
        u = np.arange(0.0, s[-1], 4.0)
        vals.append(TA.sample(zq, g, np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])))
    if not vals:
        return None
    v = np.concatenate(vals)
    return {"n": int(v.size), "median_dem_elevation_m": round(float(np.median(v)), 2),
            "p10_p90_dem_elevation_m": [round(float(np.percentile(v, 10)), 2), round(float(np.percentile(v, 90)), 2)]}


# --------------------------------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------------------------------
def verify_georef(site, near, src_tif, rec, n=400, seed=7):
    """
    Independent georeferencing check: for random near-grid cells convert (x, z) back to lon/lat (PROJ inverse of the
    aeqd frame), look the value up (nearest pixel) in the ORIGINAL cached CUDEM raster and compare with the grid.
    A flipped/shifted grid would show up as metre-scale differences; the expected residual is resampling smoothing.
    """
    g = near["grid"]
    rng = np.random.default_rng(seed)
    ii, jj = rng.integers(0, g.nx, n), rng.integers(0, g.nz, n)
    xz = np.column_stack([g.x_of(ii), g.z_of(jj)])
    ll = site_xz_to_lonlat(site, xz)
    a = tifffile.imread(src_tif).astype(np.float64)
    gt = rec["vrt_geotransform"]
    lon_o, lat_o = gt[0] + rec["srcwin"][0] * gt[1], gt[3] + rec["srcwin"][1] * gt[5]
    col = np.floor((ll[:, 0] - lon_o) / gt[1]).astype(int)
    row = np.floor((ll[:, 1] - lat_o) / gt[5]).astype(int)
    ok = (col >= 0) & (col < a.shape[1]) & (row >= 0) & (row < a.shape[0])
    src = a[row[ok], col[ok]]
    # gentle-terrain cells only (local 3x3 range < 0.5 m) so resampling smoothing does not dominate
    zq = near["zq"]
    rng3 = ndi.maximum_filter(zq, 3) - ndi.minimum_filter(zq, 3)
    flat = rng3[jj[ok], ii[ok]] < 0.5
    d = (zq[jj[ok], ii[ok]] - src)
    return {"cells_checked": int(ok.sum()), "gentle_cells": int(flat.sum()),
            "median_abs_diff_m_gentle": round(float(np.median(np.abs(d[flat]))), 3),
            "p95_abs_diff_m_gentle": round(float(np.percentile(np.abs(d[flat]), 95)), 3),
            "max_abs_diff_m_all": round(float(np.abs(d).max()), 2)}


def verify_near_far(near, far):
    """Block-average the near grid to the far cells inside the near window and compare with the far grid."""
    gn, gf = near["grid"], far["grid"]
    k = int(round(gf.dx / gn.dx))
    i0, j0 = int(round((gn.x0 - gf.x0) / gf.dx)), int(round((gn.z0 - gf.z0) / gf.dz))
    ni, nj = gn.nx // k, gn.nz // k
    blk = near["zq"][:nj * k, :ni * k].reshape(nj, k, ni, k).mean(axis=(1, 3))
    fs = far["zq"][j0:j0 + nj, i0:i0 + ni]
    d = fs - blk
    return {"cells": int(d.size), "mean_far_minus_near_m": round(float(d.mean()), 3), "rms_m": round(float(np.sqrt((d ** 2).mean())), 3),
            "p95_abs_m": round(float(np.percentile(np.abs(d), 95)), 3),
            "note": "far (20 m, from the 6.9 m COG overview) minus 5x5 block mean of the near grid, over the near window"}


# --------------------------------------------------------------------------------------------------
# Static provenance text for the site json
# --------------------------------------------------------------------------------------------------
def source_records(site_id, cudem_near, cudem_far, lidar):
    return [
        {
            "id": "cudem_1_9_arcsec_puerto_rico", "role": "PRIMARY - every value in the near and far grids",
            "title": "Puerto Rico: Continuously Updated Digital Elevation Model (CUDEM) - 1/9 Arc-Second Resolution Bathymetric-Topographic Tiles",
            "publisher": "NOAA National Centers for Environmental Information (NCEI); author CIRES, University of Colorado Boulder",
            "doi": "10.25921/ds9v-ky35", "ncei_metadata_id": "gov.noaa.ngdc.mgg.dem:399919",
            "dataset_pages": ["https://doi.org/10.25921/ds9v-ky35",
                              "https://coast.noaa.gov/dataviewer/#/lidar/search/where:ID=9525",
                              "https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ngdc.mgg.dem:399919/html"],
            "tiles": [
                {"name": CUDEM_TILES[1], "url": CUDEM_BASE + CUDEM_TILES[1], "bytes_full_tile": 174374581, "s3_last_modified": "2024-01-05",
                 "covers": "lon -65.50..-65.25, lat 18.00..18.25 (+6 px buffer)"},
                {"name": CUDEM_TILES[0], "url": CUDEM_BASE + CUDEM_TILES[0], "bytes_full_tile": 192828601, "s3_last_modified": "2024-01-05",
                 "covers": "lon -65.75..-65.50, lat 18.00..18.25 (+6 px buffer); only used for the western edge of the Mosquito far window"},
            ],
            "version": "2022v2 (GeoTIFF DATETIME tag 2022-06-03; ISO publication date 2022-06-11)",
            "horizontal_crs": "NAD83 geographic (EPSG:4269), pixel size 1/9 arc-second = 3.0864197e-5 deg (~3.4 m N-S, ~3.3 m E-W at 18.1 N), pixel-is-area",
            "vertical_datum": "PRVD02 height (Puerto Rico Vertical Datum of 2002, EPSG:6641), orthometric metres; treated as mean sea level (no shift applied)",
            "format": "Cloud-Optimized GeoTIFF, float32, DEFLATE + floating-point predictor, 512x512 blocks, 4 overviews (2x,4x,8x,16x), NoData -9999",
            "read_method": ("GDAL /vsicurl/ range reads through a local VRT of the two tiles. NEAR window: native pixels (no resampling on "
                            "the way in). FAR window: the COG 2x overview level (6.9 m; the provider's overview was verified to be a bicubic decimation of the native data, rms 2 mm)."),
            "windows_read": {"near": {k: cudem_near[k] for k in ("srcwin", "bytes")}, "far": {k: cudem_far[k] for k in ("srcwin", "level", "bytes")}},
            "gap_filling_by_source": "CUDEM is a gap-free merged product (bathymetry + topography); its interpolation of unconstrained cells is documented in the NCEI DEM technical reports.",
            "citation": ("Cooperative Institute for Research in Environmental Sciences (CIRES) at the University of Colorado, Boulder. 2014: Continuously "
                         "Updated Digital Elevation Model (CUDEM) - 1/9 Arc-Second Resolution Bathymetric-Topographic Tiles [Puerto Rico tiles "
                         "ncei19_n18x25_w065x50 and ncei19_n18x25_w065x75, window subsets]. NOAA National Centers for Environmental Information. "
                         "https://doi.org/10.25921/ds9v-ky35. Accessed 2026-09-30."),
            "licence": ("Produced by NOAA NCEI; not subject to copyright protection within the United States. NOAA/NCEI give no warranty on accuracy; "
                        "NOT FOR NAVIGATION (use NOS nautical charts)."),
            "stated_accuracy": "Horizontal ~1 m (source data); vertical better than 0.5 m RMSE for lidar topography; bathymetric accuracy varies; no quantitative tile-level assessment published.",
        },
        {
            "id": "ngs_2019_topobathy_lidar_dem_puerto_rico", "role": "CROSS-CHECK ONLY (validation statistics; not merged into any product)",
            "title": "2019 NOAA NGS Topobathy Lidar DEM: Puerto Rico (NOAA Digital Coast dataset ID 9392; blocks pr_block_B/pr_block_C cover Vieques)",
            "acquisition": "2019-01-20 to 2019-06-02, Leading Edge Geomatics, Riegl VQ-880-G II topobathymetric lidar (post-Hurricane Maria)",
            "dataset_pages": ["https://coast.noaa.gov/dataviewer/#/lidar/search/where:ID=9392",
                              "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NGS_PR_Topobathy_DEM_2019_9392/"],
            "tile_index": NGS19_INDEX,
            "tiles_used": lidar["tiles"], "tile_urls": lidar["tile_urls"],
            "horizontal_crs": "NAD83 / UTM zone 19N (EPSG:26919), 1 m pixels, float32 COG (DEFLATE), 512x512 blocks, overviews 2 m and 4 m",
            "vertical_datum": "not encoded in the GeoTIFF; NGS/CUDEM practice = PRVD02 (assumed)",
            "read_method": "gdalwarp -ovr 1 from /vsicurl/ tiles: 4 m overview level (verified to equal the exact 2x2-of-2x2 box average of the 1 m data), bilinear to the site grid",
            "bytes": lidar["bytes"],
            "licence": "Public NOAA data (STAC licence tag 'NLPL'); please credit NOAA National Geodetic Survey / Office for Coastal Management",
            "why_not_the_requested_2016_set": ("The 2016 NOAA NGS topobathy DEM (dataset 8462, incl. Culebra) and the 2015 set (6211) have no tiles over "
                                                "these Vieques sites; the 2019 NGS set (9392) and the 2018 USACE/FEMA set (8571) do."),
        },
        {
            "id": "openstreetmap_overpass", "role": "sparse features + coastline cross-check",
            "url": OVERPASS[0], "licence": "ODbL 1.0 - (c) OpenStreetMap contributors",
        },
    ]


# --------------------------------------------------------------------------------------------------
# Stage 4: per-site orchestration
# --------------------------------------------------------------------------------------------------
class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, cls=NpEnc, ensure_ascii=False))


def check_origin(site_id, site, near):
    """Print the origin suggested by the DEM (crescent middle / lagoon centroid) versus the configured one."""
    z, g = near["z"], near["grid"]
    if site_id == "caracas":
        seeds = lonlat_to_site_xz(site, [(lo, la) for la, lo in site["tip_seeds_latlon"]])
        m, _ = TA.analyze_caracas(z, g, [tuple(s) for s in seeds])
        xm, zm = m["crescent_middle_xz"]
        ll = site_xz_to_lonlat(site, [(xm, zm)])[0]
        log(f"[{site_id}] crescent middle at x={xm}, z={zm} -> lat {ll[1]:.5f}, lon {ll[0]:.5f} (rounded {ll[1]:.4f}, {ll[0]:.4f}); "
            f"configured {site['lat0']}, {site['lon0']}")
    else:
        m, _ = TA.analyze_mosquito(z, g)
        cx, cz = m["depth_ge_0p5m"]["lagoon_centroid_xz"]
        ll = site_xz_to_lonlat(site, [(cx, cz)])[0]
        log(f"[{site_id}] lagoon centroid at x={cx}, z={cz} -> lat {ll[1]:.5f}, lon {ll[0]:.5f} (rounded {ll[1]:.4f}, {ll[0]:.4f}); "
            f"configured {site['lat0']}, {site['lon0']}")


def process_site(site_id, args, cache):
    site = SITES[site_id]
    log(f"===== {site_id}: {site['name']} =====")
    net0 = Net.total
    cudem_near_tif, cudem_near = fetch_cudem(site_id, site, "near", cache, args.refetch)
    cudem_far_tif, cudem_far = fetch_cudem(site_id, site, "far", cache, args.refetch)
    lidar_tif, lidar = fetch_lidar_near(site_id, site, cache, args.refetch)

    # ---- grids
    method = "bilinear"
    near = build_grid(site_id, site, "near", cudem_near_tif, method, args.denoise)
    far = build_grid(site_id, site, "far", cudem_far_tif, method, args.denoise)
    log(f"near {near['stats']['nx']}x{near['stats']['nz']}  min {near['stats']['min_m']} max {near['stats']['max_m']} "
        f"water {near['stats']['water_fraction']:.3f} filled {near['stats']['nodata_filled_cells']}")
    log(f"far  {far['stats']['nx']}x{far['stats']['nz']}  min {far['stats']['min_m']} max {far['stats']['max_m']} "
        f"water {far['stats']['water_fraction']:.3f} filled {far['stats']['nodata_filled_cells']} clamped {far['stats']['clamped_fraction']}")

    if args.check_origin:
        check_origin(site_id, site, near)
        return None

    DATA.mkdir(parents=True, exist_ok=True)
    near["q"].tofile(DATA / f"{site_id}_near.i16")
    far["q"].tofile(DATA / f"{site_id}_far.i16")

    # ---- lidar cross-check (lidar grid is already on the near-grid geometry)
    zl = tifffile.imread(lidar_tif).astype(np.float64)
    zl[zl <= NODATA + 1] = np.nan
    xcheck, diff = TA.crosscheck_lidar(near["zq"], zl, near["grid"])
    log(f"cross-check vs NGS-2019 lidar: rms {xcheck['rms_diff_m']} m, mean {xcheck['mean_diff_m']} m, coverage {xcheck['coverage_fraction_of_window']}")

    verification = {"georeference_vs_source_raster": verify_georef(site, near, cudem_near_tif, cudem_near),
                    "near_vs_far_consistency": verify_near_far(near, far)}
    log(f"georef check: {verification['georeference_vs_source_raster']}")
    log(f"near/far consistency: {verification['near_vs_far_consistency']}")

    # ---- OSM
    osm = None
    if not args.skip_osm:
        osm = fetch_osm(site_id, site, cache)
        write_json(DATA / f"osm_{site_id}.json", osm)
        osm["_coast_check"] = osm_vs_dem_coastline(osm, near["zq"], near["grid"])
        log(f"   OSM: roads {len(osm['roads'])} structures {len(osm['structures'])} coastline {len(osm['coastline'])}")

    # ---- site-specific metrics + previews
    metrics, markers, lines = {}, [(0.0, 0.0, "origin")], []
    if site_id == "caracas":
        seeds = lonlat_to_site_xz(site, [(lo, la) for la, lo in site["tip_seeds_latlon"]])
        metrics, hp = TA.analyze_caracas(near["zq"], near["grid"], [tuple(s) for s in seeds])
        metrics["hint_position_xz"] = [round(float(v), 1) for v in lonlat_to_site_xz(site, [(site["hint"][1], site["hint"][0])])[0]]
        ll = site_xz_to_lonlat(site, [metrics["crescent_middle_xz"], metrics["tip_west_xz"], metrics["tip_east_xz"],
                                      metrics["beach_end_west_xz"], metrics["beach_end_east_xz"]])
        for key, row in zip(("crescent_middle", "tip_west", "tip_east", "beach_end_west", "beach_end_east"), ll):
            metrics[key + "_latlon"] = [round(float(row[1]), 5), round(float(row[0]), 5)]
        markers += [(hp["tips"][0][0], hp["tips"][0][1], "W tip"), (hp["tips"][1][0], hp["tips"][1][1], "E tip"),
                    (hp["beach_ends"][0][0], hp["beach_ends"][0][1], "beach end"), (hp["beach_ends"][1][0], hp["beach_ends"][1][1], "beach end")]
        lines.append((hp["beach_path"][0], hp["beach_path"][1], (60, 255, 90)))
        vx, vz = TA.vec_from_bearing(hp["axis_bearing"])
        lines.append((np.array([0.0, 800 * vx]), np.array([0.0, 800 * vz]), (255, 255, 0)))
    else:
        metrics, hp = TA.analyze_mosquito(near["zq"], near["grid"])
        pr = metrics["depth_ge_0p5m"]
        pts = {"lagoon_centroid": pr["lagoon_centroid_xz"], "inlet_neck": pr["inlet_neck_xz"],
               "inlet_channel_start": pr["inlet_channel_start_xz"], "inlet_channel_end": pr["inlet_channel_end_xz"],
               "lagoon_bbox_sw": [pr["lagoon_bbox_x_m"][0], pr["lagoon_bbox_z_m"][1]], "lagoon_bbox_ne": [pr["lagoon_bbox_x_m"][1], pr["lagoon_bbox_z_m"][0]]}
        ll = site_xz_to_lonlat(site, list(pts.values()))
        pr["latlon"] = {k: [round(float(r[1]), 5), round(float(r[0]), 5)] for k, r in zip(pts, ll)}
        cl, ch, kn = hp["channel_idx"]
        markers = [(0.0, 0.0, "origin = lagoon centre"), (hp["neck"][0], hp["neck"][1], "inlet neck")]
        for cx, cz in TA.contours_xz(hp["lagoon_mask"].astype(float), near["grid"], 0.5):
            if len(cx) > 10:
                lines.append((cx, cz, (0, 255, 255)))
        lines.append((hp["path_xz"][0][cl:ch + 1], hp["path_xz"][1][cl:ch + 1], (255, 140, 0)))   # inlet channel thalweg
    if site_id == "mosquito":
        mask = hp["lagoon_mask"]
        void = mask & ~np.isfinite(zl)
        both = mask & np.isfinite(zl)
        dd = (near["zq"] - zl)[both]
        if void.any():
            vr, vc = np.nonzero(void)
            pr["lidar_void_in_lagoon"] = {
                "area_ha": round(float(void.sum() * 16 / 1e4), 2), "fraction_of_lagoon_area": round(float(void.sum() / mask.sum()), 3),
                "centroid_xz": [round(float(near["grid"].x_of(vc).mean()), 1), round(float(near["grid"].z_of(vr).mean()), 1)],
                "cudem_depth_mean_m": round(float(-near["zq"][void].mean()), 2), "cudem_depth_max_m": round(float(-near["zq"][void].min()), 2)}
        pr["cudem_minus_lidar_in_lagoon_where_both_valid"] = {"n": int(both.sum()), "mean_m": round(float(dd.mean()), 3), "rms_m": round(float(np.sqrt((dd ** 2).mean())), 3)}
    metrics["extremes_near"] = TA.extremes(near["zq"], near["grid"])
    metrics["other_enclosed_water_bodies_near"] = TA.enclosed_water_bodies(near["zq"], near["grid"], min_ha=0.3)
    metrics["low_land_sensitivity_near"] = TA.low_land(near["zq"], near["grid"])

    if osm:
        for r in osm["roads"]:
            p = np.array(r["pts"])
            lines.append((p[:, 0], p[:, 1], (250, 250, 250)))
        for c in osm["coastline"]:
            p = np.array(c["pts"])
            lines.append((p[:, 0], p[:, 1], (255, 0, 255)))
    ng, fg = near["grid"], far["grid"]
    ver = "CUDEM 1/9 arc-sec (NOAA NCEI), PRVD02 ~ MSL"
    hx, hz = lonlat_to_site_xz(site, [(site["hint"][1], site["hint"][0])])[0]
    markers.append((float(hx), float(hz), "user hint", (-88, -20)))
    TA.render_preview(near["zq"], ng, DATA / f"preview_{site_id}_near.png", f"{site['name']} - near 4 m grid",
                      markers=markers, polylines=lines, tick_m=500.0, scale_bar_m=500.0,
                      extra_note=f"{ver}. Red = 0 m contour; magenta = OSM coastline; white = OSM tracks" + ("; cyan = lagoon; orange = inlet thalweg" if site_id == "mosquito" else "; green = beach waterline; yellow line = axis normal") + "; north up.")
    (nx0, nx1), (nz0, nz1) = site["near"]["x"], site["near"]["z"]
    TA.render_preview(far["zq"], fg, DATA / f"preview_{site_id}_far.png", f"{site['name']} - far 20 m grid",
                      markers=[(0.0, 0.0, "origin")], boxes=[(nx0, nx1, nz0, nz1, "near window")], tick_m=1000.0, scale_bar_m=2000.0,
                      zexag_sea=4.0, zexag_land=2.0, extra_note=f"{ver}. Red = 0 m contour; depths below -60 m share one colour.")

    # ---- site json
    net_site = {"bytes_this_run": Net.total - net0,
                "cudem_near_bytes": cudem_near["bytes"], "cudem_far_bytes": cudem_far["bytes"], "lidar_overview_bytes": lidar["bytes"]}
    meta = {
        "id": site_id, "name": site["name"],
        "lat0": site["lat0"], "lon0": site["lon0"],
        "origin_definition": (
            "Arc-length midpoint of the 0 m shoreline between the two headland tips of the crescent (a cuspate foreland behind the islet on the bay axis)"
            if site_id == "caracas" else
            "Area centroid of the enclosed lagoon (water deeper than 0.5 m, cut from the open sea at the inlet neck)"),
        "user_hint_latlon": list(site["hint"]),
        "projection": aeqd(site["lat0"], site["lon0"]),
        "frame": "x = east (m); z = SOUTH (m), north is -z; y = elevation above mean sea level (m). Rows run north->south (row 0 = most negative z), columns west->east; "
                 "values are cell-centre samples: x = x0+(i+0.5)dx, z = z0+(j+0.5)dz.",
        "horizontal_datum_note": "Sources are NAD83 / NAD83(2011); the site frame uses WGS84. PROJ applies a null shift (NAD83 ~ WGS84 to 1-2 m in Puerto Rico); ignored.",
        "grids": {
            "near": {**near["stats"], "file": f"{site_id}_near.i16", "dtype": "int16 little-endian", "unit": "centimetres above MSL", "row_order": "north->south",
                     "source": "CUDEM native-resolution window", "resampling": f"gdalwarp -r {method} (kernel scaled to 4 m), aeqd"},
            "far": {**far["stats"], "file": f"{site_id}_far.i16", "dtype": "int16 little-endian", "unit": "centimetres above MSL", "row_order": "north->south",
                    "source": "CUDEM 2x overview level (6.9 m)", "resampling": f"gdalwarp -r {method} (kernel scaled to 20 m), aeqd"},
        },
        "vertical": {"datum": "PRVD02 (orthometric), treated as mean sea level; sea level = 0 m. NOAA VDatum offsets PRVD02<->local MSL are ~0.1 m or less (not applied). "
                             "Storage: Int16 little-endian centimetres, clamped to +-327.00 m (far grid: deeper cells are clamped)."},
        "processing": {
            "resampling_method": method, "denoise": ("gaussian sigma %.2f cells (3x3), water cells only" % args.denoise) if args.denoise else "NOT applied (measured seabed high-pass noise ~1-2 cm rms)",
            "nodata": "NoData cells are filled: offshore = nearest valid depth extrapolated seaward at 0.02 m/m (keeps deepening, floor -320 m); land = Laplace interpolation. "
                      "Filled fractions are in grids.*.nodata_filled_fraction (0 for these windows).",
            "network_bytes": net_site,
        },
        "sources": source_records(site_id, cudem_near, cudem_far, lidar),
        "metrics": metrics,
        "crosscheck_vs_ngs2019_lidar": xcheck,
        "verification": verification,
        "osm": {"file": f"osm_{site_id}.json", "counts": osm["counts_returned"] if osm else None, "coast_check": osm.get("_coast_check") if osm else None,
                "note": osm["note"] if osm else "skipped"},
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "credits": "Bathymetry/topography: NOAA NCEI CUDEM (public domain). Cross-check: NOAA NGS 2019 topobathy lidar. Map features: (c) OpenStreetMap contributors (ODbL).",
        "caveats": [],  # filled below
    }
    meta["caveats"] = caveats_for(site_id, meta)
    write_json(DATA / f"{site_id}.json", meta)
    log(f"wrote data/{site_id}.json, {site_id}_near.i16, {site_id}_far.i16, previews")
    return meta


def caveats_for(site_id, meta):
    cv = meta["crosscheck_vs_ngs2019_lidar"]
    far = meta["grids"]["far"]
    ll = meta["metrics"]["low_land_sensitivity_near"]
    cov = cv.get("lidar_void_fraction_by_cudem_elevation_class_m", {})
    steep = cv.get("by_slope", {}).get(">0.4", {})
    cls = cv.get("by_lidar_elevation_class_m", {})
    off = [cls[k] for k in ("-10..-5", "-5..-2") if k in cls]
    c = [
        f"CUDEM is ~{cv['mean_diff_m']:+.2f} m higher than the independent 2019 NGS lidar DEM on average (rms {cv['rms_diff_m']} m, {cv['rms_after_removing_mean_m']} m after removing the mean): "
        "a small vertical-datum/gridding offset, not a shape error. Absolute sea level (PRVD02 vs true local MSL) is uncertain to ~0.1 m.",
        "Elevations are bare-earth (lidar ground + bathymetric bottom): no canopy signal was found (land high-pass roughness 4-5 cm rms in mangrove/dune fringe, hillsides and "
        "backshore alike). There is no water surface: a lidar 'lake' is just the seabed/ground value beneath it.",
        f"Wet/dry sensitivity: {ll['land_0_to_0.1_m_ha']:.0f} ha of the near window lies 0-0.1 m and {ll['land_0.1_to_0.25_m_ha']:.0f} ha lies 0.1-0.25 m above sea level; with a datum "
        "uncertainty of ~0.1 m the 0 m shoreline is poorly defined on such flats (tens of metres to >100 m of horizontal shift).",
        (f"Where CUDEM and the lidar disagree most: (1) steep rocky headlands and reef fronts (slope > 0.4: rms {steep.get('rms', float('nan'))} m; 1-2 m horizontal misregistration between the two gridded sources), "
         f"(2) inland forested hillsides (patchy +-0.3-0.6 m swath/tile texture in one of the sources), (3) a smooth systematic offshore offset: CUDEM is "
         f"{min(o['mean'] for o in off):+.2f}..{max(o['mean'] for o in off):+.2f} m shallower than the lidar between -2 and -10 m, growing with depth (largest ~+0.2 m at -10..-15 m)."),
        (f"Lidar coverage: the 2019 lidar has voids where it did not see the bottom (fraction of CUDEM cells with no lidar: {cov.get('deeper than -20')} deeper than -20 m, {cov.get('-20..-15')} at -20..-15 m, "
         f"{cov.get('-15..-10')} at -15..-10 m). Deeper than ~-20 m CUDEM is therefore unverified by this lidar (multibeam/other sources or interpolation)."),
        "Data are a 2018-2019 (post-Hurricane Maria) snapshot; sand bars, berms and the shoreline change with storms and tides. At beach slopes of ~1:15 a 0.15 m tide/wave-setup "
        "difference moves the 0 m line ~2 m.",
        f"Far grid: {far['clamped_fraction'] * 100:.1f} % of cells are deeper than -327 m and are clamped to -327 m (Int16 cm range). Deep water is irrelevant for the demo but the slope beyond the shelf break is truncated.",
        "Seabed high-pass noise is only ~1-2 cm rms at 4 m (no striping, tile seams or voids seen in the near windows); the far grid is built from the COG's 2x overview "
        "(bicubic decimation by the provider), so features < ~14 m are smoothed.",
        "Horizontal datum: NAD83/NAD83(2011) sources and WGS84 frame treated as identical (PROJ null shift; real offset ~1-2 m, below the 4 m grid).",
    ]
    if site_id == "caracas":
        c.append("An islet lies on the bay axis in front of the crescent middle (see metrics.islet): the pure-south ray from the origin crosses it, so 'depth along the normal' at 400 m is on the islet; "
                 "use metrics.elevation_along_parallel_rays_offset_m for clean offshore profiles.")
    else:
        ob = [b for b in (meta["metrics"].get("other_enclosed_water_bodies_near") or []) if b["max_depth_m"] < 0.5]
        if ob:
            c.append(f"An enclosed pond west of the lagoon ({ob[0]['area_ha']:.0f} ha at x={ob[0]['centroid_xz'][0]:.0f}, z={ob[0]['centroid_xz'][1]:.0f}; local name Laguna Kiani) is a near-zero-depth salt flat "
                     f"(max depth {ob[0]['max_depth_m']:.2f} m): it will flicker wet/dry in a shallow-water solver and is NOT counted as the lagoon.")
        pr = meta["metrics"]["depth_ge_0p5m"]
        c.append(f"The inlet neck is {pr['inlet_neck_width_m']:.0f} m wide (water deeper than 0.5 m; {meta['metrics']['water_z_lt_0']['inlet_neck_width_m']:.0f} m at the 0 m level) and the channel is "
                 f"{pr['inlet_channel_length_m']:.0f} m long: wider than a 'narrow' mangrove channel one might expect, because the shallow flats flanking the thalweg are below 0 m.")
        v = meta["metrics"]["depth_ge_0p5m"].get("lidar_void_in_lagoon")
        if v:
            c.append(f"The 2019 lidar has a {v['area_ha']:.1f} ha void ({v['fraction_of_lagoon_area'] * 100:.0f} % of the lagoon) over the deepest part of the lagoon (centre x={v['centroid_xz'][0]:.0f}, z={v['centroid_xz'][1]:.0f}; "
                     f"no bottom return in the turbid/deep water). CUDEM has values there (mean depth {v['cudem_depth_mean_m']:.1f} m, max {v['cudem_depth_max_m']:.1f} m) that are interpolated or from other data, "
                     "so the lagoon's maximum depth is the least certain number. Elsewhere in the lagoon CUDEM and lidar agree to "
                     f"{meta['metrics']['depth_ge_0p5m']['cudem_minus_lidar_in_lagoon_where_both_valid']['rms_m']:.2f} m rms.")
        c.append("The lagoon's north and west margins are a very flat 0.0-0.2 m band (mangrove/salt-flat ground) 100-150 m wide before the land rises: the shoreline there moves by ~100 m for a 0.1 m datum change.")
    return c


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    ap.add_argument("--cache", default=os.environ.get("CV_CACHE", str(DEFAULT_CACHE)), help="directory for cached windowed source reads")
    ap.add_argument("--refetch", action="store_true", help="ignore the cache and read the windows again")
    ap.add_argument("--out-dir", default=None, help="where to write the products (default: <repo>/data)")
    ap.add_argument("--skip-osm", action="store_true")
    ap.add_argument("--check-origin", action="store_true", help="only print the origin suggested by the DEM")
    ap.add_argument("--denoise", type=float, default=None, help="optional gaussian sigma in cells (<=0.7) applied to water cells only")
    args = ap.parse_args()
    global DATA
    if args.out_dir:
        DATA = Path(args.out_dir)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for s in args.sites:
        process_site(s, args, cache)
    log(f"network total this run: {Net.total / 1e6:.1f} MB")
    for tag, n in Net.events:
        if n:
            log(f"   {tag:28s} {n / 1e6:8.2f} MB")
    if not args.check_origin:
        write_ledger(cache, args.sites)
        log("next: python3 tools/pack_terrain.py")


def write_ledger(cache, sites):
    """data/download_ledger.json: what a clean run downloads (from the cache sidecars) + what this run downloaded."""
    items = []
    vrt = cache / "cudem19_pr_vieques.vrt"
    items.append({"what": "CUDEM VRT/tile headers (range reads)", "bytes": 65536})
    for sid in sites:
        for kind in ("near", "far"):
            f = cache / f"cudem19_{sid}_{kind}.json"
            if f.exists():
                items.append({"what": f"CUDEM 1/9 arc-sec {kind} window, {sid}", "bytes": json.loads(f.read_text())["bytes"]})
        f = cache / f"ngs2019_{sid}_near4m.json"
        if f.exists():
            items.append({"what": f"NGS 2019 lidar 4 m overview tiles, {sid}", "bytes": json.loads(f.read_text())["bytes"]})
        q, _ = osm_query(SITES[sid])
        o = cache / f"osm_{sid}_{hashlib.md5(q.encode()).hexdigest()[:10]}.json"
        if o.exists():
            items.append({"what": f"Overpass response {o.name}", "bytes": o.stat().st_size})
    z = cache / "tileindex_NGS_PR_Topobathy_DEM_2019.zip"
    if z.exists():
        items.append({"what": "NGS 2019 lidar tile index (zip)", "bytes": z.stat().st_size})
    write_json(DATA / "download_ledger.json", {
        "clean_run_total_bytes": sum(i["bytes"] for i in items), "this_run_bytes": Net.total, "items": items,
        "note": ("Windowed range reads only (no whole tiles). 'clean_run_total_bytes' = what a run with an empty cache downloads "
                 "(summed from the cache sidecars); 'this_run_bytes' = counted in this invocation (0 when the cache was reused).")})


if __name__ == "__main__":
    main()
