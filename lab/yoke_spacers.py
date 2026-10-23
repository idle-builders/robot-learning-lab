"""Candidate fusion of existing spacer rings into their mating printed yoke.

The fused candidate preserves the nominal material region; linked carriers add
explicit connecting webs. The linked option is selectable in experimental
configurations. Joint interfaces are unchanged; print and strength are unqualified.
"""

from lab.leg_mechanism import yoke_parts


def select_spacers(spacers, mode):
    """Select generic yoke spacers; gripper fingers retain their loose rings."""
    if mode == "loose_rings":
        return spacers
    if mode != "linked_carriers":
        raise ValueError("unknown yoke spacer mode")
    from build123d import Compound

    return Compound(linked_spacers(spacers))


def integrate_spacers(yoke, spacers):
    rings = list(spacers.solids())
    if len(rings) != 8:
        raise ValueError("Expected eight separate spacer rings")
    fused = yoke.fuse(*rings)
    if not fused.is_valid or len(fused.solids()) != 1:
        raise ValueError("Spacers must join the yoke into one valid solid")
    if abs(fused.volume - yoke.volume - sum(r.volume for r in rings)) > 0.01:
        raise ValueError("Integration changes occupied material volume")
    return fused


def candidate_parts():
    yoke, spacers, screws = yoke_parts()
    return yoke, spacers, screws, integrate_spacers(yoke, spacers)


def linked_spacers(spacers):
    """Two four-ring carriers; thin annular web lies against each yoke face.

    Fixed study dimensions: web radii 6.8/8 mm, 0.5 mm thick. These add stock
    only near the plate side, preserving original ring bearing faces and bores.
    """
    from build123d import Pos, Cylinder

    carriers = []
    for sign in (-1, 1):
        rings = [s for s in spacers.solids() if s.center().Z * sign > 0]
        if len(rings) != 4:
            raise ValueError("Each carrier requires four original rings")
        web = Pos(0, 0, sign * 20.0) * (Cylinder(8, 0.5) - Cylinder(6.8, 1))
        joined = web.fuse(*rings)
        # The connecting web crosses the bolt circle; preserve every original bore.
        for x, y in [(-7, 0), (7, 0), (0, -7), (0, 7)]:
            joined -= Pos(x, y, sign * 19.4375) * Cylinder(1.65, 3)
        if not joined.is_valid or len(joined.solids()) != 1:
            raise ValueError("Four-ring carrier must be one connected solid")
        carriers.append(joined)
    return carriers
