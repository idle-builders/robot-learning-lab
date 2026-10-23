"""Subtractive clamp study; preserves seating skin and bolt-surround stock.

Separate experiment geometry, not an enabled robot configuration or a strength
qualification. The existing installed clamp remains the source of truth.
"""

from build123d import Axis, Box, Compound, Cylinder, Pos, Rot

from .joint_module import joint_parts


def rounded_box(x, y, z, axis, radius):
    box = Box(x, y, z)
    return box.fillet(radius, box.edges().filter_by(axis))


def candidates():
    source = joint_parts(clamped=True)
    result = {}
    # Clamped coordinates, shared with the original C018 study: inner seat
    # +/-12.365 mm and exterior +/-15.365 mm. No case, clamp or bolt moves.
    for name, sign in (("clamp_lower", -1), ("clamp_upper", 1)):
        original = source[name]
        # The cutter extends 0.1 mm beyond the exterior: actual recess depth
        # is 1.5 mm, leaving 1.5 mm of the original 3 mm back wall.
        recess = Pos(-12.5, sign * 14.665, 0) * rounded_box(34, 1.6, 18, Axis.Y, 2)
        ears = Compound(
            [
                Pos(x, sign * 6.15, 0) * rounded_box(8, 12.1, 8, Axis.X, 1.5)
                for x in (-38.615, 13.615)
            ]
        )
        seat = Pos(-12.5, sign * 12.965, 0) * Box(46.03, 1.2, 28)
        bolt_stock = Compound(
            [
                Pos(x, 0, z) * Rot(90, 0, 0) * Cylinder(3.65, 40)
                for x in (-38.615, 13.615)
                for z in (-8, 8)
            ]
        )
        result[name] = dict(
            original=original,
            recessed=original - recess,
            windowed=(original - recess) - ears,
            seating_witness=Compound(original.intersect(seat)),
            bolt_stock_witness=Compound(original.intersect(bolt_stock)),
        )
    return result
