"""Official Camera Module 3 Wide CAD, registered to the optical-front datum."""

from functools import lru_cache
import hashlib
import math
from pathlib import Path
from build123d import import_step, Pos, Rot, Compound, GeomType


def registration(raw, transform):
    substrate = max(raw.solids(), key=lambda shape: shape.volume)
    pcb = transform * substrate
    found = []
    for edge in pcb.edges():
        if edge.geom_type == GeomType.CIRCLE and abs(edge.radius - 1.1) < 1e-4:
            point = list(edge.arc_center)[1:]
            if not any(math.dist(point, p) < 1e-4 for p in found):
                found.append(point)
    expected = [[-10.5, -12.4], [10.5, -12.4], [-10.5, 0.1], [10.5, 0.1]]
    residuals = [min((math.dist(p, e) for p in found), default=1000) for e in expected]
    lens = transform * min(raw.solids(), key=lambda shape: shape.bounding_box().min.Z)
    bounds = lens.bounding_box()
    optical = [
        bounds.max.X,
        (bounds.min.Y + bounds.max.Y) / 2,
        (bounds.min.Z + bounds.max.Z) / 2,
    ]
    return dict(
        status="pass"
        if len(found) == 4
        and max(residuals) < 1e-4
        and math.dist(optical, [0, 0, 0]) < 1e-4
        else "fail",
        hole_centers_yz_mm=found,
        hole_residuals_mm=residuals,
        lens_front_center_mm=optical,
        scope="BRep hole-circle and front mechanical lens-center registration; not optical principal-plane calibration",
    )


@lru_cache(maxsize=2)
def _load(path, digest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("camera vendor geometry changed")
    raw = import_step(path)
    transform = Pos(-8.805, -12.5, -14.4) * Rot(0, -90, 0)
    registered = registration(raw, transform)
    negative = registration(raw, Pos(0, 1, 0) * transform)
    shapes = [transform * shape for shape in raw.solids()]
    invalid = [i for i, shape in enumerate(shapes) if not shape.is_valid]
    if (
        len(shapes) != 631
        or invalid
        or registered["status"] != "pass"
        or negative["status"] != "fail"
    ):
        raise ValueError(f"camera vendor validity/registration failed: {invalid}")
    module = Compound(children=shapes)
    return module, dict(
        solid_count=len(shapes),
        invalid_solids=invalid,
        registration=registered,
        displaced_registration_negative_control=negative,
        source_sha256=digest,
        scope="vendor purchased assembly, not one connected printed part; physical revision unconfirmed",
    )


def model(spec):
    # Recheck content even when the expensive imported geometry is cached.
    path, digest = spec["source_vendor"], spec["vendor_sha256"]
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("camera vendor geometry changed")
    return _load(path, digest)


def assembly_overlap(a, b):
    """Cull only disjoint vendor solids before the same exact Boolean operation.

    Purchased assemblies can contain hundreds of solids. No bounding-box volume
    substitutes for geometry, and no touching/overlapping boxes are discarded.
    """
    from .packaging import overlap as exact_overlap

    aa, bb = a.bounding_box(), b.bounding_box()

    def intersects(x, y):
        return all(
            min(hi, hj) >= max(lo, lj)
            for hi, hj, lo, lj in zip(x.max, y.max, x.min, y.min)
        )

    if not intersects(aa, bb):
        return 0.0
    sa, sb = list(a.solids()), list(b.solids())
    if len(sa) > 32:
        sa = [p for p in sa if intersects(p.bounding_box(), bb)]
    if len(sb) > 32:
        sb = [p for p in sb if intersects(p.bounding_box(), aa)]
    if not sa or not sb:
        return 0.0
    return exact_overlap(Compound(sa), Compound(sb))
