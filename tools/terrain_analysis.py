#!/usr/bin/env python3
"""
terrain_analysis.py - validation helpers for the ClearVieques terrain builder.

Everything here works on grids in the *site frame* used by build_terrain.py:

    x = east (m), z = SOUTH (m; north is -z), y = elevation above (PRVD02 ~ MSL) in m.
    Arrays are row-major, row 0 = northernmost, column 0 = westernmost, and cell (row j, col i)
    has its centre at  x = x0 + (i + 0.5) * dx,  z = z0 + (j + 0.5) * dz.

Contents
    Grid                  tiny grid-geometry helper (index <-> metric coordinates, bilinear sampling)
    hillshade/ramp/...    preview rendering (hillshade blended with a bathymetric colour ramp + 0 m contour)
    analyze_caracas       crescent-beach metrics (headland tips, beach length/slope, shore normal, offshore depths)
    analyze_mosquito      lagoon / inlet metrics (neck location and width, lagoon extent and depth statistics)
    crosscheck_lidar      CUDEM vs lidar difference statistics and hot spots

Dependencies: numpy, scipy (ndimage), scikit-image (measure.find_contours, graph.route_through_array), Pillow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

try:  # scikit-image is only needed for contour extraction / least-cost paths
    from skimage import graph as skgraph
    from skimage import measure as skmeasure
except ImportError as exc:  # pragma: no cover
    raise SystemExit("terrain_analysis.py needs scikit-image (pip install scikit-image)") from exc


# --------------------------------------------------------------------------------------------------
# Grid geometry
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    nx: int
    nz: int
    dx: float
    dz: float
    x0: float  # west edge of column 0 (m, east positive)
    z0: float  # north edge of row 0 (m, south positive => usually negative)

    def x_of(self, col):
        return self.x0 + (np.asarray(col, float) + 0.5) * self.dx

    def z_of(self, row):
        return self.z0 + (np.asarray(row, float) + 0.5) * self.dz

    def col_of(self, x):
        return (np.asarray(x, float) - self.x0) / self.dx - 0.5

    def row_of(self, z):
        return (np.asarray(z, float) - self.z0) / self.dz - 0.5

    def contains(self, x, z):
        return (self.x0 <= x < self.x0 + self.nx * self.dx) and (self.z0 <= z < self.z0 + self.nz * self.dz)

    @property
    def extent(self):
        """(xmin, xmax, zmin, zmax) of the outer cell edges."""
        return self.x0, self.x0 + self.nx * self.dx, self.z0, self.z0 + self.nz * self.dz


def sample(z, g: Grid, x, zq, order=1):
    """Bilinear (order=1) sample of grid `z` at metric positions (x, zq); clamps at the borders."""
    x = np.asarray(x, float)
    zq = np.asarray(zq, float)
    coords = np.array([g.row_of(zq).ravel(), g.col_of(x).ravel()])
    out = ndi.map_coordinates(z, coords, order=order, mode="nearest")
    return out.reshape(x.shape)


def bearing_deg(dx_east, dz_south):
    """Compass bearing (deg clockwise from north) of a vector given in the site frame (x east, z south)."""
    return (math.degrees(math.atan2(dx_east, -dz_south)) + 360.0) % 360.0


def vec_from_bearing(b_deg):
    """Unit vector (x, z) for a compass bearing."""
    b = math.radians(b_deg)
    return math.sin(b), -math.cos(b)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------------------------------
# Contours / polylines
# --------------------------------------------------------------------------------------------------
def contours_xz(z, g: Grid, level=0.0):
    """All iso-lines of `level` as list of (x[], z[]) arrays in metres (marching squares, linear interp)."""
    out = []
    for c in skmeasure.find_contours(z, level):
        out.append((g.x_of(c[:, 1]), g.z_of(c[:, 0])))
    return out


def arclen(x, z):
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])


def resample_path(x, z, step=2.0):
    s = arclen(x, z)
    su = np.arange(0.0, s[-1], step)
    return np.interp(su, s, x), np.interp(su, s, z), su


def smooth_path(x, z, sigma_m, step=2.0):
    """Uniform resample + gaussian smoothing of a polyline; returns xs, zs, tangent (unit), arclen."""
    xu, zu, su = resample_path(x, z, step)
    s = max(sigma_m / step, 0.5)
    xs = ndi.gaussian_filter1d(xu, s, mode="nearest")
    zs = ndi.gaussian_filter1d(zu, s, mode="nearest")
    tx, tz = np.gradient(xs, step), np.gradient(zs, step)
    n = np.hypot(tx, tz)
    n[n == 0] = 1.0
    return xs, zs, tx / n, tz / n, su


# --------------------------------------------------------------------------------------------------
# Preview rendering
# --------------------------------------------------------------------------------------------------
SEA_RAMP = [  # depth (m, negative) -> RGB ; shallow = pale turquoise, deep = navy
    (-200, (4, 12, 52)), (-80, (6, 26, 90)), (-40, (10, 55, 130)), (-20, (18, 95, 168)),
    (-10, (30, 140, 200)), (-5, (62, 182, 216)), (-2, (120, 214, 224)), (-0.5, (176, 234, 230)),
    (0, (206, 244, 236)),
]
LAND_RAMP = [  # elevation (m) -> RGB ; sand -> green -> olive -> brown
    (0, (238, 222, 172)), (1.5, (216, 208, 150)), (6, (150, 184, 100)), (25, (86, 142, 70)),
    (60, (96, 122, 66)), (120, (140, 118, 84)), (300, (170, 150, 120)),
]


def _ramp(stops, v):
    xs = np.array([s[0] for s in stops], float)
    cs = np.array([s[1] for s in stops], float)
    out = np.empty(v.shape + (3,), float)
    for k in range(3):
        out[..., k] = np.interp(v, xs, cs[:, k])
    return out


def hillshade(z, dx, dz, azimuth=315.0, altitude=42.0, zexag=1.0):
    gy, gx = np.gradient(z * zexag, dz, dx)  # gy: d/d(south), gx: d/d(east)
    dzde, dzdn = gx, -gy
    az, alt = math.radians(azimuth), math.radians(altitude)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    hs = (-dzde * lx - dzdn * ly + lz) / np.sqrt(dzde ** 2 + dzdn ** 2 + 1.0)
    return np.clip(hs, 0.0, 1.0)


def _font(size):
    for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_preview(z, g: Grid, path, title, zexag_land=2.5, zexag_sea=6.0, scale=None, markers=(),
                   polylines=(), boxes=(), tick_m=500.0, scale_bar_m=500.0, extra_note=""):
    """
    Hillshade blended with a bathymetry/land colour ramp, 0 m contour in red, metric axes (x east, z south),
    north arrow (north is UP), scale bar and a depth legend. `markers` = [(x, z, label[, (dx, dy) label offset in px])], `polylines` =
    [(xs, zs, (r,g,b))], `boxes` = [(x0,x1,z0,z1,label)] (e.g. the near window drawn on the far preview).
    """
    z = np.asarray(z, float)
    sea = z < 0.0
    rgb = np.where(sea[..., None], _ramp(SEA_RAMP, np.minimum(z, 0.0)), _ramp(LAND_RAMP, np.maximum(z, 0.0)))
    hs = np.where(sea, hillshade(z, g.dx, g.dz, zexag=zexag_sea), hillshade(z, g.dx, g.dz, zexag=zexag_land))
    rgb = np.clip(rgb * (0.50 + 0.65 * hs)[..., None], 0, 255).astype(np.uint8)

    if scale is None:
        scale = 1.5 if g.nx <= 1000 else 1.0
        if g.nx <= 700:
            scale = 2.0
    im = Image.fromarray(rgb).resize((int(g.nx * scale), int(g.nz * scale)), Image.BICUBIC)
    W, H = im.size
    d = ImageDraw.Draw(im)
    px = lambda x: (np.asarray(x, float) - g.x0) / g.dx * scale          # noqa: E731
    py = lambda zz: (np.asarray(zz, float) - g.z0) / g.dz * scale        # noqa: E731

    # 0 m contour
    for cx, cz in contours_xz(z, g, 0.0):
        if len(cx) < 4:
            continue
        pts = list(zip(px(cx).tolist(), py(cz).tolist()))
        d.line(pts, fill=(226, 32, 32), width=max(1, int(round(scale))))
    for xs_, zs_, col in polylines:
        d.line(list(zip(px(xs_).tolist(), py(zs_).tolist())), fill=col, width=2)
    for (bx0, bx1, bz0, bz1, label) in boxes:
        d.rectangle([px(bx0), py(bz0), px(bx1), py(bz1)], outline=(255, 255, 255), width=2)
        d.text((px(bx0) + 4, py(bz0) + 3), label, fill=(255, 255, 255), font=_font(13))
    f = _font(14)
    for mk in markers:                       # (x, z, label) or (x, z, label, (dx_px, dy_px)) label offset
        mx, mz, label = mk[:3]
        ox, oy = mk[3] if len(mk) > 3 else (9, 4)
        X, Y = float(px(mx)), float(py(mz))
        d.ellipse([X - 6, Y - 6, X + 6, Y + 6], outline=(255, 255, 0), width=2)
        d.line([X - 10, Y, X + 10, Y], fill=(255, 255, 0), width=1)
        d.line([X, Y - 10, X, Y + 10], fill=(255, 255, 0), width=1)
        d.text((X + ox, Y + oy), label, fill=(255, 255, 0), font=f)

    # canvas with margins for axes
    ml, mt, mr, mb = 62, 64, 92, 44
    canvas = Image.new("RGB", (W + ml + mr, H + mt + mb), (250, 250, 248))
    canvas.paste(im, (ml, mt))
    c = ImageDraw.Draw(canvas)
    ft, fs = _font(16), _font(12)
    c.text((ml, 8), title, fill=(20, 20, 20), font=ft)
    if extra_note:
        c.text((ml, 29), extra_note, fill=(70, 70, 70), font=fs)
    c.rectangle([ml, mt, ml + W, mt + H], outline=(60, 60, 60))
    xmin, xmax, zmin, zmax = g.extent
    t = math.ceil(xmin / tick_m) * tick_m
    while t <= xmax + 1e-6:
        X = ml + (t - g.x0) / g.dx * scale
        c.line([X, mt + H, X, mt + H + 5], fill=(60, 60, 60))
        c.text((X - 14, mt + H + 7), f"{t:+.0f}", fill=(40, 40, 40), font=fs)
        t += tick_m
    t = math.ceil(zmin / tick_m) * tick_m
    while t <= zmax + 1e-6:
        Y = mt + (t - g.z0) / g.dz * scale
        c.line([ml - 5, Y, ml, Y], fill=(60, 60, 60))
        c.text((6, Y - 6), f"{t:+.0f}", fill=(40, 40, 40), font=fs)
        t += tick_m
    c.text((ml + W // 2 - 40, mt + H + 24), "x east (m)", fill=(40, 40, 40), font=fs)
    c.text((4, mt - 16), "z south (m)", fill=(40, 40, 40), font=fs)
    # north arrow (north is up in every preview)
    ax, ay = ml + W - 28, mt + 46
    d2 = ImageDraw.Draw(canvas)
    d2.polygon([(ax, ay - 30), (ax - 8, ay), (ax + 8, ay)], fill=(255, 255, 255), outline=(20, 20, 20))
    d2.text((ax - 5, ay + 2), "N", fill=(255, 255, 255), font=ft)
    # scale bar
    L = scale_bar_m / g.dx * scale
    sx, sy = ml + 14, mt + H - 18
    d2.rectangle([sx - 4, sy - 16, sx + L + 4, sy + 10], fill=(255, 255, 255))
    d2.line([sx, sy, sx + L, sy], fill=(0, 0, 0), width=3)
    d2.text((sx, sy - 14), f"{scale_bar_m:.0f} m", fill=(0, 0, 0), font=fs)
    # legend
    lx0, lx1 = ml + W + 16, ml + W + 32
    ly0, ly1 = mt + 10, mt + H - 10
    hgt = ly1 - ly0
    zmax_leg, zmin_leg = 60.0, -60.0
    for k in range(hgt):
        v = zmax_leg + (zmin_leg - zmax_leg) * k / (hgt - 1)
        col = _ramp(LAND_RAMP, np.array([max(v, 0.0)]))[0] if v >= 0 else _ramp(SEA_RAMP, np.array([v]))[0]
        c.line([lx0, ly0 + k, lx1, ly0 + k], fill=tuple(int(u) for u in col))
    for v in (60, 20, 5, 0, -5, -20, -60):
        Y = ly0 + (zmax_leg - v) / (zmax_leg - zmin_leg) * (hgt - 1)
        c.line([lx1, Y, lx1 + 4, Y], fill=(60, 60, 60))
        c.text((lx1 + 6, Y - 6), f"{v:+d}", fill=(40, 40, 40), font=fs)
    c.text((lx0 - 2, ly0 - 16), "z (m)", fill=(40, 40, 40), font=fs)
    canvas.save(path)
    return canvas.size


# --------------------------------------------------------------------------------------------------
# Caracas: crescent beach metrics
# --------------------------------------------------------------------------------------------------
def _outward_normals(xs, zs, tx, tz, z, g: Grid, probe=30.0):
    """Unit normals of a smoothed shoreline path, oriented seaward (DEM < 0 at probe distance)."""
    nx_, nz_ = -tz, tx  # one of the two candidates
    a = sample(z, g, xs + probe * nx_, zs + probe * nz_)
    flip = np.where(a < 0, 1.0, -1.0)  # want sea (negative) on the +n side
    # decide orientation for the path as a whole (robust to isolated wrong picks)
    if np.median(flip) < 0:
        nx_, nz_ = -nx_, -nz_
    return nx_, nz_


def _mainland_contour(z, g: Grid):
    cs = contours_xz(z, g, 0.0)
    cs.sort(key=lambda c: -arclen(*c)[-1])
    x, zz = cs[0]
    if x[0] > x[-1]:
        x, zz = x[::-1], zz[::-1]
    return x, zz


def analyze_caracas(z, g: Grid, tip_seeds_xz, origin_hint=(0.0, 0.0), log=print):
    """
    Crescent metrics from the near grid. Returns (metrics dict, drawing helpers dict).

    Definitions (also written to the site json):
      * headland tips  = seaward-most (max z, i.e. southernmost) points of the mainland 0 m contour within 250 m
                         of the two seed positions flanking the crescent.
      * crescent shoreline = 0 m contour between the two tips; 'crescent middle' = its arc-length midpoint.
      * axis normal   = perpendicular to the tip-to-tip chord, pointing seaward.
      * local normal  = seaward normal of the shoreline smoothed over +-100 m at the crescent middle.
      * beach         = contiguous stretch of shoreline around the middle whose seaward normal stays within
                         +-70 deg of the axis normal (i.e. excluding the sides of the headlands).
      * beach slope   = 2 m / horizontal distance between the +1 m and -1 m contours, measured along shore-normal
                         transects every 6 m along the beach.
    """
    m = {}
    x, zz = _mainland_contour(z, g)
    s_raw = arclen(x, zz)

    # tips
    tips = []
    for (sx, sz) in tip_seeds_xz:
        d = np.hypot(x - sx, zz - sz)
        idx = np.nonzero(d < 250.0)[0]
        i = idx[np.argmax(zz[idx])]
        tips.append(i)
    iw, ie = sorted(tips)  # polyline runs west -> east, so the lower index is the western tip
    tw, te = (x[iw], zz[iw]), (x[ie], zz[ie])
    m["tip_west_xz"] = [round(float(tw[0]), 1), round(float(tw[1]), 1)]
    m["tip_east_xz"] = [round(float(te[0]), 1), round(float(te[1]), 1)]
    chord = (te[0] - tw[0], te[1] - tw[1])
    m["tip_chord_length_m"] = round(math.hypot(*chord), 1)

    # crescent shoreline & arc-length midpoint
    seg_x, seg_z = x[iw:ie + 1], zz[iw:ie + 1]
    seg_s = arclen(seg_x, seg_z)
    half = seg_s[-1] / 2.0
    xm, zm = float(np.interp(half, seg_s, seg_x)), float(np.interp(half, seg_s, seg_z))
    m["shoreline_length_between_tips_m"] = round(float(seg_s[-1]), 1)
    m["crescent_middle_xz"] = [round(xm, 1), round(zm, 1)]
    m["crescent_middle_offset_from_origin_m"] = round(math.hypot(xm - origin_hint[0], zm - origin_hint[1]), 1)
    m["origin_elevation_m"] = round(float(sample(z, g, origin_hint[0], origin_hint[1])), 2)

    # axis normal = chord rotated to point seaward (south side)
    cxn, czn = -chord[1], chord[0]
    if czn < 0:
        cxn, czn = -cxn, -czn
    nrm = math.hypot(cxn, czn)
    cxn, czn = cxn / nrm, czn / nrm
    axis_b = bearing_deg(cxn, czn)
    m["axis_normal_bearing_deg"] = round(axis_b, 1)

    # smoothed shoreline between the tips -> local normals
    xs, zs, tx, tz, su = smooth_path(seg_x, seg_z, sigma_m=100.0)
    nxs, nzs = _outward_normals(xs, zs, tx, tz, z, g, probe=40.0)
    k = int(np.argmin(np.hypot(xs - xm, zs - zm)))
    local_b = bearing_deg(nxs[k], nzs[k])
    m["local_normal_bearing_deg"] = round(local_b, 1)

    # beach extent (normal within +-70 deg of the axis normal, contiguous around the middle)
    xs2, zs2, tx2, tz2, su2 = smooth_path(seg_x, seg_z, sigma_m=40.0)
    n2x, n2z = _outward_normals(xs2, zs2, tx2, tz2, z, g, probe=40.0)
    ang = np.array([wrap180(bearing_deg(a, b) - axis_b) for a, b in zip(n2x, n2z)])
    ok = np.abs(ang) <= 70.0
    k2 = int(np.argmin(np.hypot(xs2 - xm, zs2 - zm)))
    lo = k2
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = k2
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    m["beach_length_m"] = round(float(su2[hi] - su2[lo]), 1)
    m["beach_end_west_xz"] = [round(float(xs2[lo]), 1), round(float(zs2[lo]), 1)]
    m["beach_end_east_xz"] = [round(float(xs2[hi]), 1), round(float(zs2[hi]), 1)]
    m["beach_chord_length_m"] = round(math.hypot(xs2[hi] - xs2[lo], zs2[hi] - zs2[lo]), 1)

    # slope transects along the beach (raw contour resolution, normals from a 15 m smoothed path)
    xs3, zs3, tx3, tz3, su3 = smooth_path(seg_x, seg_z, sigma_m=15.0)
    n3x, n3z = _outward_normals(xs3, zs3, tx3, tz3, z, g, probe=40.0)
    # restrict to beach stretch (map by nearest arclength position)
    i_lo = int(np.argmin(np.hypot(xs3 - xs2[lo], zs3 - zs2[lo])))
    i_hi = int(np.argmin(np.hypot(xs3 - xs2[hi], zs3 - zs2[hi])))
    t = np.arange(-90.0, 90.0 + 0.25, 0.5)
    slopes, widths = [], []
    for i in range(min(i_lo, i_hi), max(i_lo, i_hi) + 1, 3):  # every 6 m (path step 2 m)
        prof = sample(z, g, xs3[i] + t * n3x[i], zs3[i] + t * n3z[i])
        j0 = int(np.argmin(np.abs(t)))
        # landward: first t<=0 where profile >= +1 ; seaward: first t>=0 where profile <= -1
        land_idx = np.nonzero(prof[:j0 + 1][::-1] >= 1.0)[0]
        sea_idx = np.nonzero(prof[j0:] <= -1.0)[0]
        if len(land_idx) == 0 or len(sea_idx) == 0:
            continue
        t_p1 = t[j0 - land_idx[0]]
        t_m1 = t[j0 + sea_idx[0]]
        w = t_m1 - t_p1
        if w <= 0:
            continue
        slopes.append(2.0 / w)
        widths.append(w)
    slopes, widths = np.array(slopes), np.array(widths)
    if len(slopes):
        m["beach_slope_mean_tan"] = round(float(slopes.mean()), 4)
        m["beach_slope_median_tan"] = round(float(np.median(slopes)), 4)
        m["beach_slope_p10_p90_tan"] = [round(float(np.percentile(slopes, 10)), 4), round(float(np.percentile(slopes, 90)), 4)]
        m["beach_slope_from_mean_width_tan"] = round(float(2.0 / widths.mean()), 4)
        m["beach_slope_mean_1_in"] = round(float(1.0 / slopes.mean()), 1)
        m["beach_face_width_m_mean_minus1_to_plus1"] = round(float(widths.mean()), 1)
        m["beach_slope_transects_used"] = int(len(slopes))
    else:
        m["beach_slope_mean_tan"] = None

    # offshore depth along normals (from the origin/crescent middle)
    radii = [25, 50, 100, 200, 400, 800]
    o = origin_hint
    for name, b in (("axis", axis_b), ("local", local_b), ("south", 180.0)):
        vx, vz = vec_from_bearing(b)
        vals = [float(sample(z, g, o[0] + r * vx, o[1] + r * vz)) for r in radii]
        m[f"elevation_along_{name}_normal_m"] = {str(r): round(v, 2) for r, v in zip(radii, vals)}
    # lateral offsets around the islet: rays parallel to the axis normal starting 200/300 m west/east of the origin
    vx, vz = vec_from_bearing(axis_b)
    px_, pz_ = vz, -vx  # unit vector perpendicular to the axis, pointing east-ish (+x when the axis points south)
    lat = {}
    for off in (-300.0, -200.0, 200.0, 300.0):
        ox, oz = o[0] + off * px_, o[1] + off * pz_
        lat[f"{off:+.0f}"] = {str(r): round(float(sample(z, g, ox + r * vx, oz + r * vz)), 2) for r in radii}
    m["elevation_along_parallel_rays_offset_m"] = lat

    # islet in front of the beach: largest land component not connected to the mainland
    land = z >= 0.0
    lab, n = ndi.label(land)
    r0, c0 = int(g.row_of(-900.0)), int(g.col_of(origin_hint[0]))
    main_lab = lab[max(r0, 0), c0] if lab[max(r0, 0), c0] > 0 else np.bincount(lab[lab > 0]).argmax()
    best = None
    for L in range(1, n + 1):
        if L == main_lab:
            continue
        mask = lab == L
        area = mask.sum() * g.dx * g.dz
        if area < 5000:
            continue
        rr, cc = np.nonzero(mask)
        xc, zc = float(g.x_of(cc).mean()), float(g.z_of(rr).mean())
        if zc < origin_hint[1] and best is not None:
            continue
        d = math.hypot(xc - origin_hint[0], zc - (origin_hint[1] + 300.0))
        if best is None or d < best[0]:
            best = (d, L, area, xc, zc, mask)
    if best is not None:
        _, L, area, xc, zc, mask = best
        rr, cc = np.nonzero(mask)
        m["islet"] = {
            "centroid_xz": [round(xc, 1), round(zc, 1)],
            "x_range_m": [round(float(g.x_of(cc.min()) - g.dx / 2), 1), round(float(g.x_of(cc.max()) + g.dx / 2), 1)],
            "z_range_m": [round(float(g.z_of(rr.min()) - g.dz / 2), 1), round(float(g.z_of(rr.max()) + g.dz / 2), 1)],
            "area_m2": round(float(area), 0),
            "max_elevation_m": round(float(z[mask].max()), 1),
        }

    helpers = {
        "shore_x": seg_x, "shore_z": seg_z, "tips": [tw, te], "middle": (xm, zm),
        "beach_ends": [(xs2[lo], zs2[lo]), (xs2[hi], zs2[hi])], "beach_path": (xs2[lo:hi + 1], zs2[lo:hi + 1]),
        "axis_bearing": axis_b,
    }
    return m, helpers


# --------------------------------------------------------------------------------------------------
# Mosquito: lagoon / inlet metrics
# --------------------------------------------------------------------------------------------------
def _bottleneck(dist, seed_a, seed_b):
    """
    Widest-path ('bottleneck') value between two seed pixels of a distance-transform map: the largest r such that
    the set {dist >= r} still connects the seeds. Returns r (in cells).
    """
    lo, hi = 0.0, float(min(dist[seed_a], dist[seed_b]))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lab, _ = ndi.label(dist >= mid)
        if lab[seed_a] > 0 and lab[seed_a] == lab[seed_b]:
            lo = mid
        else:
            hi = mid
    return lo


def analyze_mosquito(z, g: Grid, origin=(0.0, 0.0), thr_depth=0.5, channel_width_m=200.0, log=print):
    """
    Enclosed-lagoon metrics, computed for two water definitions: depth >= thr_depth (primary) and z < 0.

      * thalweg  = least-cost path (through the widest connection) from the lagoon core (max inscribed circle within
                   300 m of the origin) to the open-sea core (max inscribed circle in the southern 15 % of the window).
      * neck     = narrowest point of that path; inlet width = 2 x the inscribed radius there (bottleneck of the
                   Euclidean distance transform between lagoon and sea).
      * channel  = contiguous thalweg stretch around the neck whose local width (2 x inscribed radius) < channel_width_m.
      * lagoon   = water body that survives an opening by channel_width_m/2 (i.e. everything wider than the channel),
                   grown back by the same radius inside the water mask - so the inlet channel is excluded.
    """
    m = {}
    dxm = g.dx
    R = channel_width_m / 2.0
    rr, cc = np.mgrid[0:g.nz, 0:g.nx]
    X, Zc = g.x_of(cc), g.z_of(rr)
    res = {}
    for label, thr in (("depth_ge_0p5m", -thr_depth), ("water_z_lt_0", 0.0)):
        water = z < thr
        dist = ndi.distance_transform_edt(water) * dxm  # metres to nearest non-water cell
        a_idx = np.unravel_index(np.argmax(np.where((np.hypot(X - origin[0], Zc - origin[1]) < 300.0) & water, dist, -1)), dist.shape)
        south = Zc > (g.z0 + 0.85 * g.nz * g.dz)
        b_idx = np.unravel_index(np.argmax(np.where(south & water, dist, -1)), dist.shape)
        r_neck = _bottleneck(dist, a_idx, b_idx)

        # thalweg through the widest connection (restricted to {dist >= r_neck}, which is connected by construction)
        cost = np.where(dist >= r_neck - 1e-6, 1.0 / (dist + 1.0) ** 2, 1e9)
        path, _ = skgraph.route_through_array(cost, a_idx, b_idx, fully_connected=True, geometric=True)
        path = np.array(path)
        pw = 2.0 * dist[path[:, 0], path[:, 1]]                       # local channel width along the thalweg (m)
        step = np.concatenate([[0.0], np.cumsum(np.hypot(*(np.diff(path, axis=0) * dxm).T))])
        k0 = int(np.argmin(pw))
        lo = hi = k0
        while lo > 0 and pw[lo - 1] <= pw[k0] + 6.0:                  # run of near-minimal width around the minimum
            lo -= 1
        while hi < len(pw) - 1 and pw[hi + 1] <= pw[k0] + 6.0:
            hi += 1
        kn = (lo + hi) // 2
        neck = (float(g.x_of(path[kn, 1])), float(g.z_of(path[kn, 0])))
        cl, ch = kn, kn                                               # channel extent (local width < channel_width_m)
        while cl > 0 and pw[cl - 1] < channel_width_m:
            cl -= 1
        while ch < len(pw) - 1 and pw[ch + 1] < channel_width_m:
            ch += 1

        # lagoon body: opening by R, geodesic growth back by R inside the water mask
        core_lab, _ = ndi.label(dist >= R)
        core = core_lab == core_lab[a_idx]
        d2core = ndi.distance_transform_edt(~core) * dxm
        lag = water & (d2core <= R + 0.5 * dxm)
        lab_l, _ = ndi.label(lag)
        lagoon = lab_l == lab_l[a_idx]
        r_, c_ = np.nonzero(lagoon)
        depth = -z[lagoon]
        xl, zl = g.x_of(c_), g.z_of(r_)
        area = lagoon.sum() * g.dx * g.dz
        pts = np.column_stack([xl - xl.mean(), zl - zl.mean()])
        _, vec = np.linalg.eigh(np.cov(pts.T))
        proj = pts @ vec
        ext = proj.max(axis=0) - proj.min(axis=0)
        thal_depth = -z[path[lo:hi + 1, 0], path[lo:hi + 1, 1]]
        res[label] = {
            "threshold_m": thr,
            "lagoon_centroid_xz": [round(float(xl.mean()), 1), round(float(zl.mean()), 1)],
            "lagoon_bbox_x_m": [round(float(xl.min() - g.dx / 2), 1), round(float(xl.max() + g.dx / 2), 1)],
            "lagoon_bbox_z_m": [round(float(zl.min() - g.dz / 2), 1), round(float(zl.max() + g.dz / 2), 1)],
            "lagoon_area_ha": round(float(area / 1e4), 1),
            "lagoon_major_length_m": round(float(ext[1]), 0), "lagoon_minor_width_m": round(float(ext[0]), 0),
            "depth_mean_m": round(float(depth.mean()), 2),
            "depth_median_m": round(float(np.median(depth)), 2),
            "depth_max_m": round(float(depth.max()), 2),
            "max_depth_xz": [round(float(xl[np.argmax(depth)]), 1), round(float(zl[np.argmax(depth)]), 1)],
            "volume_m3": round(float(depth.sum() * g.dx * g.dz), 0),
            "inlet_neck_xz": [round(neck[0], 1), round(neck[1], 1)],
            "inlet_neck_width_m": round(float(2.0 * r_neck), 1),
            "inlet_neck_thalweg_depth_m": round(float(np.median(thal_depth)), 2),
            "inlet_channel_start_xz": [round(float(g.x_of(path[cl, 1])), 1), round(float(g.z_of(path[cl, 0])), 1)],
            "inlet_channel_end_xz": [round(float(g.x_of(path[ch, 1])), 1), round(float(g.z_of(path[ch, 0])), 1)],
            "inlet_channel_length_m": round(float(step[ch] - step[cl]), 0),
            "inlet_channel_width_range_m": [round(float(pw[cl:ch + 1].min()), 0), round(float(pw[cl:ch + 1].max()), 0)],
            "inlet_channel_definition": f"thalweg stretch around the neck where local width (2 x inscribed radius) < {channel_width_m:.0f} m",
            "lagoon_definition": f"water body that survives an opening by {R:.0f} m radius, grown back by {R:.0f} m inside the water mask",
        }
        res[label]["_mask"] = lagoon
        res[label]["_path_xz"] = (g.x_of(path[:, 1]), g.z_of(path[:, 0]))
        res[label]["_channel"] = (cl, ch, kn)
    m["primary"] = "depth_ge_0p5m"
    for k, v in res.items():
        m[k] = {a: b for a, b in v.items() if not a.startswith("_")}
    pri = res["depth_ge_0p5m"]
    helpers = {"lagoon_mask": pri["_mask"], "path_xz": pri["_path_xz"], "channel_idx": pri["_channel"],
               "neck": tuple(pri["inlet_neck_xz"]), "centroid": tuple(pri["lagoon_centroid_xz"])}
    return m, helpers


def extremes(z, g: Grid):
    """Locations of the global minimum / maximum elevation of a grid (site frame metres)."""
    j, i = np.unravel_index(np.argmax(z), z.shape)
    j2, i2 = np.unravel_index(np.argmin(z), z.shape)
    return {"max_elevation_m": round(float(z[j, i]), 2), "max_elevation_xz": [round(float(g.x_of(i)), 1), round(float(g.z_of(j)), 1)],
            "min_elevation_m": round(float(z[j2, i2]), 2), "min_elevation_xz": [round(float(g.x_of(i2)), 1), round(float(g.z_of(j2)), 1)]}


def enclosed_water_bodies(z, g: Grid, min_ha=0.5, thr=0.0, top=6):
    """
    Water bodies (z < thr) that are NOT connected to the largest (open-sea) water component and do not touch the window
    edge (a body touching the edge may connect to the sea outside the window, so it is not reported as enclosed).
    """
    water = z < thr
    lab, n = ndi.label(water)
    if n == 0:
        return []
    sizes = ndi.sum(water, lab, range(1, n + 1))
    sea = int(np.argmax(sizes)) + 1
    edge_labels = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))) - {0}
    out = []
    for L in range(1, n + 1):
        if L == sea or L in edge_labels:
            continue
        ha = sizes[L - 1] * g.dx * g.dz / 1e4
        if ha < min_ha:
            continue
        mask = lab == L
        rr, cc = np.nonzero(mask)
        d = -z[mask]
        out.append({"centroid_xz": [round(float(g.x_of(cc).mean()), 1), round(float(g.z_of(rr).mean()), 1)],
                    "area_ha": round(float(ha), 2), "mean_depth_m": round(float(d.mean()), 2), "max_depth_m": round(float(d.max()), 2)})
    out.sort(key=lambda o: -o["area_ha"])
    return out[:top]


def low_land(z, g: Grid):
    """Wet/dry sensitivity: how much land lies within a few decimetres above sea level (datum uncertainty ~0.1 m)."""
    cell_ha = g.dx * g.dz / 1e4
    tot = z.size * cell_ha
    out = {}
    for lo, hi in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)):
        k = (z >= lo) & (z < hi)
        out[f"land_{lo:g}_to_{hi:g}_m_ha"] = round(float(k.sum() * cell_ha), 1)
    out["window_area_ha"] = round(float(tot), 0)
    return out


# --------------------------------------------------------------------------------------------------
# CUDEM vs lidar
# --------------------------------------------------------------------------------------------------
def crosscheck_lidar(zc, zl, g: Grid, nodata=-9999.0, block_m=100.0, log=print):
    """
    Compare CUDEM (zc) with an independent lidar DEM (zl, same grid; NaN/nodata where missing).
    Returns a dict with overall and per-elevation-class statistics (CUDEM - lidar, metres) and the worst 100 m blocks.
    """
    valid = np.isfinite(zl) & (zl > nodata + 1) & np.isfinite(zc)
    d = np.where(valid, zc - zl, np.nan)
    dv = d[valid]
    out = {
        "n_cells": int(valid.sum()),
        "coverage_fraction_of_window": round(float(valid.mean()), 4),
        "mean_diff_m": round(float(dv.mean()), 3),
        "median_diff_m": round(float(np.median(dv)), 3),
        "rms_diff_m": round(float(np.sqrt((dv ** 2).mean())), 3),
        "rms_after_removing_mean_m": round(float(dv.std()), 3),
        "p2_p98_diff_m": [round(float(np.percentile(dv, 2)), 3), round(float(np.percentile(dv, 98)), 3)],
        "max_abs_diff_m": round(float(np.abs(dv).max()), 2),
    }
    classes = {}
    for lo, hi, name in [(-999, -20, "z<-20"), (-20, -10, "-20..-10"), (-10, -5, "-10..-5"), (-5, -2, "-5..-2"),
                         (-2, 0, "-2..0"), (0, 2, "0..2"), (2, 10, "2..10"), (10, 999, ">10")]:
        mk = valid & (zl >= lo) & (zl < hi)
        if mk.sum() > 50:
            dd = d[mk]
            classes[name] = {"n": int(mk.sum()), "mean": round(float(dd.mean()), 3), "rms": round(float(np.sqrt((dd ** 2).mean())), 3)}
    out["by_lidar_elevation_class_m"] = classes
    # lidar coverage (voids) by CUDEM depth class: where does the lidar stop seeing the bottom?
    cov = {}
    for lo, hi, name in [(-999, -20, "deeper than -20"), (-20, -15, "-20..-15"), (-15, -10, "-15..-10"), (-10, -5, "-10..-5"), (-5, 0, "-5..0"), (0, 999, "land")]:
        mk = np.isfinite(zc) & (zc >= lo) & (zc < hi)
        if mk.sum() > 50:
            cov[name] = round(float(1.0 - np.mean(np.isfinite(zl[mk]) & (zl[mk] > nodata + 1))), 3)
    out["lidar_void_fraction_by_cudem_elevation_class_m"] = cov
    # slope-controlled: differences on gentle terrain vs steep terrain
    gy, gx = np.gradient(np.where(np.isfinite(zl), zl, zc), g.dz, g.dx)
    slope = np.hypot(gx, gy)
    for lo, hi, name in [(0, 0.1, "slope<0.1"), (0.1, 0.4, "0.1..0.4"), (0.4, 99, ">0.4")]:
        mk = valid & (slope >= lo) & (slope < hi)
        if mk.sum() > 50:
            dd = d[mk]
            out.setdefault("by_slope", {})[name] = {"n": int(mk.sum()), "rms": round(float(np.sqrt((dd ** 2).mean())), 3), "mean": round(float(dd.mean()), 3)}
    # worst blocks
    bs = int(round(block_m / g.dx))
    nbx, nbz = g.nx // bs, g.nz // bs
    cells = []
    for bj in range(nbz):
        for bi in range(nbx):
            sl = (slice(bj * bs, (bj + 1) * bs), slice(bi * bs, (bi + 1) * bs))
            v = valid[sl]
            if v.mean() < 0.6:
                continue
            dd = d[sl][v]
            cx = float(g.x_of(bi * bs + bs / 2 - 0.5))
            cz = float(g.z_of(bj * bs + bs / 2 - 0.5))
            mean_z = float(zc[sl][v].mean())
            cells.append((float(np.sqrt((dd ** 2).mean())), cx, cz, float(dd.mean()), mean_z, float(np.abs(dd).max())))
    cells.sort(reverse=True)
    out["worst_100m_blocks"] = [
        {"centre_xz": [round(c[1]), round(c[2])], "rms_m": round(c[0], 2), "mean_diff_m": round(c[3], 2),
         "mean_cudem_elevation_m": round(c[4], 1), "max_abs_m": round(c[5], 2)} for c in cells[:6]]
    j, i = np.unravel_index(np.nanargmax(np.abs(d)), d.shape)
    out["largest_single_cell_diff"] = {"xz": [round(float(g.x_of(i))), round(float(g.z_of(j)))], "diff_m": round(float(d[j, i]), 2),
                                        "cudem_m": round(float(zc[j, i]), 2), "lidar_m": round(float(zl[j, i]), 2)}
    return out, d
