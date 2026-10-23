"""Geometric identity checks at a CAD serialization boundary (millimetres)."""

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopLoc import TopLoc_Location


def _mesh_bounds_record(
    shape,
    deflections=(0.02, 0.005, 0.001, 0.00025, 0.0000625),
    angular_tolerance=0.1,
    convergence=0.002,
):
    """Refine absolute mesh bounds to <=1 micron with a <=2 micron delta.

    A converged triangulation is numerical evidence, not an exact extrema proof.
    Clear cached meshes each time. Further refinement is bounded; failure to
    converge rejects the export rather than changing the identity tolerance.
    """
    if (
        len(deflections) < 2
        or not all(np.isfinite(d) and d > 0 for d in deflections)
        or any(b >= a for a, b in zip(deflections, deflections[1:]))
        or not np.isfinite(convergence)
        or convergence <= 0
    ):
        raise ValueError("bounds require decreasing positive finite deflections")
    previous = None
    delta = float("inf")
    for deflection in deflections:
        BRepTools.Clean_s(shape.wrapped)
        mesher = BRepMesh_IncrementalMesh(
            shape.wrapped, deflection, False, angular_tolerance, True
        )
        if not mesher.IsDone():
            raise ValueError("surface meshing failed")
        box = Bnd_Box()
        for face in shape.faces():
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
            if triangulation is None or triangulation.NbNodes() == 0:
                raise ValueError("cannot bound a face without triangulation")
            # Traverse actual transformed mesh nodes in C++; do not use cached
            # surface extrema (which underbound scaled spline ellipsoids).
            if not triangulation.MinMax(box, location.Transformation(), True):
                raise ValueError("triangulation bounds failed")
        if box.IsVoid():
            raise ValueError("cannot bound shape without triangulated faces")
        values = box.Get()
        current = np.array([values[:3], values[3:]])
        if previous is not None:
            delta = float(np.max(np.abs(current - previous)))
            if deflection <= 0.001 and delta <= convergence:
                return dict(
                    bounds_mm=current.tolist(),
                    bounds_method="absolute_deflection_mesh_convergence",
                    bounds_deflection_mm=deflection,
                    bounds_convergence_mm=convergence,
                    bounds_final_delta_mm=delta,
                )
        previous = current
    raise ValueError(
        f"surface bounds final delta {delta:.6g} mm exceeds {convergence} mm "
        "or minimum refinement was not reached"
    )


def _bounds_record(shape, **kwargs):
    # Validation meshes must not leak into STL export: downstream viewers have
    # their own deflection and triangle budget. OCCT otherwise reuses finer
    # cached meshes even when the exporter requests coarser tessellation.
    try:
        return _mesh_bounds_record(shape, **kwargs)
    finally:
        BRepTools.Clean_s(shape.wrapped)


def converged_bounds(shape, **kwargs):
    """Return bounds only; geometry_record also retains refinement evidence."""
    return _bounds_record(shape, **kwargs)["bounds_mm"]


def geometry_record(shape):
    bounds = _bounds_record(shape)
    properties = GProp_GProps()
    error = BRepGProp.VolumeProperties_s(shape.wrapped, properties, 1e-7, True, False)
    return dict(
        valid=shape.is_valid,
        volume_mm3=properties.Mass(),
        default_volume_mm3=shape.volume,
        integration="adaptive_surface_1e-7",
        estimated_relative_error=error,
        solids=len(shape.solids()),
        **bounds,
    )


def geometry_agrees(actual, expected):
    actual_volume = (
        actual["volume_mm3"]
        if expected.get("integration")
        else actual.get("default_volume_mm3", actual["volume_mm3"])
    )
    return (
        actual["valid"]
        and actual["solids"] == expected["solids"]
        and np.isclose(actual_volume, expected["volume_mm3"], rtol=0.001, atol=0.02)
        and (
            "bounds_mm" not in expected
            or np.allclose(
                actual["bounds_mm"], expected["bounds_mm"], rtol=0, atol=0.02
            )
        )
    )
