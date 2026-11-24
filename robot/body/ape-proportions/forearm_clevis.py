"""Forearm connector as a clevis on both sides of the flipped elbow servo (owner, 2026-11-13).

The owner's sketch: no open gap under the elbow servo; the forearm grips the servo on both sides and one tapered body
runs from under the servo to the wrist. Built here, right side (the left side mirrors x), heights from the elbow axis:

  - medial cheek on the horn: a 3 mm plate (x 114.2..117.2, the old horn plate) as a disc of radius CHEEK_R about the
    axis, running down to CHEEK_BOTTOM; 4 M3 clearance holes on the horn's 7 mm circle and the central screw hole;
  - lateral cheek on the rear idler: the same plate at x 155.5..158.5 with a boss down onto the idler face
    (x 154.48; the idler sits 0.2 mm proud of the case's rear face, so the plate itself stays 1 mm off the case),
    4 M3 holes on the idler's 7 mm circle and a relief for the case's 3 mm centre stub;
  - a hollow tapered body (wall WALL) lofted from the full clevis width under the cheeks to the wrist block, fused to
    the old wrist mount (the selected forearm connector below WRIST_KEEP_TOP).

The idler is the selected C047's rear idler, placed by registering the shoulder-pitch C047 (case + horn, the same
servo model) onto the elbow servo (horn centre error 0.013 mm).
"""
from __future__ import annotations

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("ape-proportions-build")   # configured values: robot/body/ape-proportions/build.bom.yaml config:

CHEEK_R = CFG["CHEEK_R"]
CHEEK_BOTTOM = CFG["CHEEK_BOTTOM"]
CHEEK_DROP_HALF = CFG["CHEEK_DROP_HALF"]
MEDIAL_X = CFG["MEDIAL_X"]
LATERAL_X = CFG["LATERAL_X"]
IDLER_FACE_X = CFG["IDLER_FACE_X"]
BOSS_R = CFG["BOSS_R"]
HOLE_R = CFG["HOLE_R"]
HOLE_PCD_R = CFG["HOLE_PCD_R"]
STUB_RELIEF_R = CFG["STUB_RELIEF_R"]
CENTRE_SCREW_R = CFG["CENTRE_SCREW_R"]
LOFT_TOP = CFG["LOFT_TOP"]
LOFT_TOP_Y = CFG["LOFT_TOP_Y"]
LOFT_TOP_X = CFG["LOFT_TOP_X"]
LOFT_BOTTOM_Z = CFG["LOFT_BOTTOM_Z"]
LOFT_BOTTOM_X = CFG["LOFT_BOTTOM_X"]
LOFT_BOTTOM_Y = CFG["LOFT_BOTTOM_Y"]
WALL = CFG["WALL"]
WRIST_KEEP_TOP = CFG["WRIST_KEEP_TOP"]
CRADLE_RELIEF_R = CFG["CRADLE_RELIEF_R"]
CRADLE_RELIEF_X = CFG["CRADLE_RELIEF_X"]


def _cyl_x(C, x0, x1, y, z, r):
    return C.cylinder((x0, y, z), (x1, y, z), r)


def cheek(C, x, o):
    """Disc about the axis plus a straight drop to CHEEK_BOTTOM (a 'tombstone' plate)."""
    disc = _cyl_x(C, x[0], x[1], o[1], o[2], CHEEK_R)
    drop = C.box((x[0], o[1] - CHEEK_DROP_HALF, o[2] + CHEEK_BOTTOM), (x[1], o[1] + CHEEK_DROP_HALF, o[2]))
    return C.fuse(disc, drop)


def holes(C, x0, x1, o, centre_r):
    out = [_cyl_x(C, x0 - 1, x1 + 1, o[1] + dy, o[2] + dz, HOLE_R)
           for dy, dz in ((HOLE_PCD_R, 0), (-HOLE_PCD_R, 0), (0, HOLE_PCD_R), (0, -HOLE_PCD_R))]
    out.append(_cyl_x(C, x0 - 1, x1 + 1, o[1], o[2], centre_r))
    return out


def tapered_body(o, sgn):
    from build123d import Rectangle, loft, offset
    x_top = sorted(sgn * x for x in LOFT_TOP_X)
    x_bot = sorted(sgn * x for x in LOFT_BOTTOM_X)
    z_top, z_bot = o[2] + LOFT_TOP, LOFT_BOTTOM_Z

    def rect(xs, ys, z):
        from build123d import Pos
        return Pos((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, z) * Rectangle(xs[1] - xs[0], ys[1] - ys[0])
    top = rect(x_top, LOFT_TOP_Y, z_top)
    bot = rect(x_bot, LOFT_BOTTOM_Y, z_bot)
    solid = loft([bot, top])
    faces = sorted(solid.faces(), key=lambda f: f.center().Z)
    tube = offset(solid, amount=-WALL, openings=[faces[0], faces[-1]])
    return tube.wrapped


def build(C, o, sgn, horn_plate, wrist_part):
    """horn_plate: the old stage 3 horn plate turned onto the forearm; wrist_part: the old forearm connector."""
    def mx(x):
        return tuple(sorted(sgn * v for v in x))
    med, lat = mx(MEDIAL_X), mx(LATERAL_X)
    o_s = np.array([o[0], o[1], o[2]])
    plate_m = C.fuse(cheek(C, med, o_s), C.common(horn_plate, C._halfspace("z", o[2] - 14.0, +1)))
    plate_l = cheek(C, lat, o_s)
    bx = sorted([sgn * IDLER_FACE_X, sgn * LATERAL_X[0]])
    boss = _cyl_x(C, bx[0], bx[1], o[1], o[2], BOSS_R)
    plate_l = C.fuse(plate_l, boss)
    plate_m = C.cut_many(plate_m, holes(C, med[0], med[1], o_s, CENTRE_SCREW_R))
    plate_l = C.cut_many(plate_l, holes(C, min(bx[0], lat[0]), max(bx[1], lat[1]), o_s, STUB_RELIEF_R))
    wrist = C.common(wrist_part, C._halfspace("z", WRIST_KEEP_TOP, -1))
    return C.fuse(tapered_body(o_s, sgn), plate_m, plate_l, wrist)


def cradle_relief(C, o, sgn):
    xs = sorted(sgn * x for x in CRADLE_RELIEF_X)
    return _cyl_x(C, xs[0], xs[1], o[1], o[2], CRADLE_RELIEF_R)


# Upper-arm column (owner, 2026-11-13: the connector should go straight down, not along the side). A hollow tube from
# the yaw-horn bracket straight down onto the elbow case clamp: round at the top, coaxial with the yaw horn so it turns
# inside the fixed shoulder cap, lofted into a rectangle centred on the case top. Open at both ends: the five yaw-horn
# screws go in from below through the tube before the elbow servo slides into the clamp and closes it.
BRACKET_KEEP_Z = CFG["BRACKET_KEEP_Z"]
COLUMN_TOP_Z = CFG["COLUMN_TOP_Z"]
COLUMN_TOP_CENTRE = CFG["COLUMN_TOP_CENTRE"]
COLUMN_TOP_R = CFG["COLUMN_TOP_R"]
COLUMN_BOTTOM_Z = CFG["COLUMN_BOTTOM_Z"]
COLUMN_BOTTOM_CENTRE = CFG["COLUMN_BOTTOM_CENTRE"]
COLUMN_BOTTOM_SIZE = CFG["COLUMN_BOTTOM_SIZE"]


def column(sgn):
    from build123d import Circle, Pos, Rectangle, loft
    cx_t, cy_t = sgn * COLUMN_TOP_CENTRE[0], COLUMN_TOP_CENTRE[1]
    cx_b, cy_b = sgn * COLUMN_BOTTOM_CENTRE[0], COLUMN_BOTTOM_CENTRE[1]

    def sec(z_b, z_t, k):
        (w, h) = COLUMN_BOTTOM_SIZE[k]
        return loft([Pos(cx_b, cy_b, z_b) * Rectangle(w, h), Pos(cx_t, cy_t, z_t) * Circle(COLUMN_TOP_R[k])])
    outer = sec(COLUMN_BOTTOM_Z, COLUMN_TOP_Z, 1)
    inner = sec(COLUMN_BOTTOM_Z - 3.0, COLUMN_TOP_Z + 0.01, 0)  # through the clamp plate, up to the bracket underside
    return outer.wrapped, inner.wrapped
