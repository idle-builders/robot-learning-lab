"""FEM library of the checks: elasticity assembly (robot/body/fem/elasticity.py), 6x6 Hooke matrices of the material table,
the gmsh tetrahedral mesher of STEP files, nodal patch weights and volume-weighted quantiles. Moved from the lower-leg
study's fem_evaluate.py without changing the maths. No dependence on the check framework, so a shim may import it."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_k] = "1"

import gmsh
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path(__import__("os").environ.get("LAB_CHECKS_DATA") or HERE / "data")


def load_elasticity():
    spec = importlib.util.spec_from_file_location("retained_elasticity", ROOT / "robot/body/fem/elasticity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ELAST = load_elasticity()
VON_MISES = ELAST.reference_namespace()["von_mises"]


def assemble_chunked(points, tet, material_matrices, cell_material_ids, chunk=40000):
    """ELAST.assemble_materials in chunks of `chunk` tets: the same stiffness (sums in a different order), but the element
    triplets of the whole mesh (3.5 GiB for the roll-flip thigh) never exist at once."""
    import scipy.sparse as sp
    ids = np.asarray(cell_material_ids, np.int32)
    K = sp.csr_matrix((3 * len(points),) * 2)
    vol = np.empty(len(tet)); B = np.empty((len(tet), 6, 12))
    for key in np.unique(ids):
        idx = np.flatnonzero(ids == key)
        ns = ELAST.reference_namespace(material_matrices[int(key)])
        for i in range(0, len(idx), chunk):
            sel = idx[i:i + chunk]
            kk, v, bb, _ = ns["assemble"](points, tet[sel])
            K = K + kk; vol[sel] = v; B[sel] = bb
            del kk
    return K, vol, B, ids


def orthotropic_matrix_MPa(Ex: float, Ey: float, Ez: float, nu: float = 0.35, G: float = 1100.0) -> np.ndarray:
    """6x6 orthotropic Hooke matrix in MPa (Voigt order: xx, yy, zz, xy, yz, xz)."""
    C = np.zeros((6, 6), float)
    C[:3, :3] = [
        [1.0 / Ex, -nu / np.sqrt(Ex * Ey), -nu / np.sqrt(Ex * Ez)],
        [-nu / np.sqrt(Ex * Ey), 1.0 / Ey, -nu / np.sqrt(Ey * Ez)],
        [-nu / np.sqrt(Ex * Ez), -nu / np.sqrt(Ey * Ez), 1.0 / Ez],
    ]
    D = np.zeros((6, 6), float)
    D[:3, :3] = np.linalg.inv(C[:3, :3])
    D[3:, 3:] = np.diag([G, G, G])
    return D


def materials():
    return json.loads((DATA / "materials.json").read_text())


def hooke(name, ln_axis, part_material=None):
    """6x6 Hooke matrix (MPa) of a material of the table; orthotropic FDM has E_layer along the layer-normal axis;
    factor_of_part_material = that factor times the part's own matrix."""
    m = materials()[name]["fem"]
    if "factor_of_part_material" in m:
        return float(m["factor_of_part_material"]) * hooke(part_material, ln_axis)
    if "E_MPa" in m:
        return ELAST.isotropic(m["E_MPa"], m["nu"])
    E = [m["E_inplane_MPa"]] * 3
    E[ln_axis] = m["E_layer_MPa"]
    return orthotropic_matrix_MPa(E[0], E[1], E[2], nu=m["nu"], G=m["G_MPa"])


def volume_quantiles(vals: np.ndarray, vols: np.ndarray) -> dict:
    if len(vals) == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    order = np.argsort(vals)
    cum = np.cumsum(vols[order])
    tot = float(cum[-1]) if cum[-1] > 0 else 1.0
    q50, q95, q99 = np.interp([0.50, 0.95, 0.99], cum / tot, vals[order])
    return {"p50": float(q50), "p95": float(q95), "p99": float(q99)}


def mesh_compound_steps(step_specs: list[tuple[str, Path]], extra_boxes_or_cyls: list[dict], h_max: float = 2.2) -> dict:
    """Mesh one or more STEP files (and optional tie primitives for multi-part baseline assemblies) into P1 tetrahedra.
    Hard limit CAD_GMSH_TIMEOUT_S (default 600 s): the process stops with its stack printed, even inside gmsh C++."""
    import faulthandler
    limit = float(os.environ.get("CAD_GMSH_TIMEOUT_S", 600))
    faulthandler.dump_traceback_later(limit, exit=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 2)
    gmsh.model.add("connector_fem")

    mat_code_map = {"metal": 0, "polymer": 1, "steel": 2}
    inputs = []
    try:
        for mat_label, step_path in step_specs:
            tags = gmsh.model.occ.importShapes(str(step_path), highestDimOnly=True)
            inputs.append((mat_label, [ (d, t) for d, t in tags if d == 3 ]))

        for prim in extra_boxes_or_cyls:
            if prim["kind"] == "cyl_z":
                tag = gmsh.model.occ.addCylinder(
                    prim["x"], prim["y"], prim["z0"], 0.0, 0.0, prim["z1"] - prim["z0"], prim["r"]
                )
                inputs.append((prim["material"], [(3, tag)]))
            elif prim["kind"] == "cyl_x":
                tag = gmsh.model.occ.addCylinder(
                    prim["x0"], prim["y"], prim["z"], prim["x1"] - prim["x0"], 0.0, 0.0, prim["r"]
                )
                inputs.append((prim["material"], [(3, tag)]))

        all_flat = [(mat, dt) for mat, dts in inputs for dt in dts]
        vol_mat = {}
        if len(all_flat) > 1:
            first_dt = [all_flat[0][1]]
            rest_dt = [dt for _, dt in all_flat[1:]]
            _, maps = gmsh.model.occ.fragment(first_dt, rest_dt)
            for (mat, _), mapping in zip(all_flat, maps):
                for dim, tag in mapping:
                    if dim == 3:
                        # Prefer steel over polymer if overlapping boundary
                        if tag not in vol_mat or mat == "steel":
                            vol_mat[tag] = mat
        else:
            vol_mat[all_flat[0][1][1]] = all_flat[0][0]

        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", max(0.35, h_max / 5.0))
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        try:
            gmsh.model.mesh.generate(3)
        except Exception:
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.20)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.model.mesh.generate(3)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        points = np.array(xyz, dtype=float).reshape(-1, 3)
        lookup = {int(t): i for i, t in enumerate(node_tags)}

        tets = []
        mat_ids = []
        for dim, tag in gmsh.model.getEntities(3):
            types, _, conn = gmsh.model.mesh.getElements(3, tag)
            m_id = mat_code_map.get(vol_mat.get(tag, "polymer"), 1)
            for typ, c in zip(types, conn):
                if typ == 4:
                    cells = np.array([lookup[int(v)] for v in c], dtype=np.int32).reshape(-1, 4)
                    tets.append(cells)
                    mat_ids.extend([m_id] * len(cells))

        tet = np.concatenate(tets, axis=0)
        mat_ids = np.array(mat_ids, dtype=np.int32)

        types2, _, conn2 = gmsh.model.mesh.getElements(2)
        tris = []
        for typ, c in zip(types2, conn2):
            if typ == 2:
                tris.append(np.array([lookup[int(v)] for v in c], dtype=np.int32).reshape(-1, 3))
        tri = np.concatenate(tris, axis=0) if tris else np.zeros((0, 3), dtype=np.int32)
    finally:
        faulthandler.cancel_dump_traceback_later()
        gmsh.finalize()

    used = np.unique(tet)
    remap = np.full(len(points), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    points = points[used]
    tet = remap[tet]
    tri = remap[tri]
    tri = tri[np.all(tri >= 0, axis=1)]

    # Remove any ultra-sliver tetrahedra if present
    xyz_t = points[tet]
    A = np.concatenate((np.ones((len(tet), 4, 1)), xyz_t), axis=2)
    vol = np.abs(np.linalg.det(A)) / 6.0
    valid_mask = vol > 1e-10
    tet = tet[valid_mask]
    mat_ids = mat_ids[valid_mask]
    vol = vol[valid_mask]

    return {
        "points": points,
        "tet": tet,
        "tri": tri,
        "mat_ids": mat_ids,
        "vol": vol,
    }


def select_patch_weights(points: np.ndarray, tri: np.ndarray, predicate) -> np.ndarray:
    """Compute lumped nodal area weights for surface triangles whose centroids satisfy `predicate(centroids, vertices)`."""
    verts = points[tri]
    cent = verts.mean(axis=1)
    area = 0.5 * np.linalg.norm(np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0]), axis=1)
    mask = predicate(cent, verts)
    if not np.any(mask):
        # Fallback to direct nodal coordinate check
        node_mask = predicate(points, points[:, None, :])
        w = np.zeros(len(points), dtype=float)
        w[node_mask] = 1.0
        return w
    w = np.bincount(tri[mask].ravel(), weights=np.repeat(area[mask] / 3.0, 3), minlength=len(points))
    return w
