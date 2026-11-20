"""Closed (watertight) surface mesh of a STEP with gmsh, for exact mesh booleans in the fit and collision checks.

OCC's per-face tessellation leaves cracks between faces of the stretched B-spline parts; gmsh meshes the shared
edges once, so the surface is conforming and closed.
"""
import numpy as np


def step_to_closed_mesh(step_path, size=0.9, timeout_s=None):
    """timeout_s (default env CAD_GMSH_TIMEOUT_S or 300): hard stop of the whole process, with the stack printed, even
    while gmsh is inside C++ (a 0.8 mm surface mesh once ran 2.4 h). Run it in a subprocess if the caller must survive.
    faulthandler has one global timer: this call replaces any timer the caller armed and cancels it on return, so it
    bounds only the gmsh call itself, not the caller's later work (2026-11-20, Idle T-be4dt7: the step-3 stall came
    after this call, in a lazy manifold3d boolean; the limit was checked to stop gmsh inside mesh.generate)."""
    import faulthandler
    import os
    import gmsh
    import trimesh
    limit = float(timeout_s or os.environ.get("CAD_GMSH_TIMEOUT_S", 300))
    print(f"gmsh surface mesh of {step_path} at {size} mm: hard limit {limit:.0f} s", flush=True)
    faulthandler.dump_traceback_later(limit, exit=True)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("s"); gmsh.model.occ.importShapes(str(step_path)); gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size); gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size * 0.15)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        idx = {int(t): i for i, t in enumerate(tags)}
        tris = []
        for et, _, nodes in zip(*gmsh.model.mesh.getElements(2)):
            if et == 2:
                tris.append(np.vectorize(idx.get)(nodes.reshape(-1, 3).astype(np.int64)))
        m = trimesh.Trimesh(np.asarray(coords).reshape(-1, 3), np.vstack(tris), process=True)
        trimesh.repair.fix_normals(m)
        return m
    finally:
        faulthandler.cancel_dump_traceback_later()
        gmsh.finalize()
