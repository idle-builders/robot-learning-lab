"""Blank-state 3D Multi-Load SIMP Topology Optimization & 3D Ground-Structure Space-Truss Discovery
for the 4-Stage Arm FDM Connector Study (`stage1_shoulder_pitch_to_roll` .. `stage4_forearm_elbow_to_wrist`).

Starts from a 100% blank state (no sheet-metal templates or reference geometry):
  1. Computes the exact 42-pose collision-free 3D flexion volume (`free_dom`) in each stage's link frame
     by boolean-subtracting the swept union of all upper-body motors, horns, screws, torso structures,
     and fitted hard-shell panels across all 42 poses (`build_poses()`).
  2. Seeds ALL 46 active physical connection points per arm (38 steel-sleeved structural bolts including
     dual-cheek / dual-face C047 motor case PA2 mounts + 8 integrated captive-M3 hard-shell mounts) and
     carves passive void channels along all straight-line screw/hex-driver insertion rays.
  3. Runs both:
     - 3D Multi-Load Orthotropic SIMP Topology Optimization (`solve_simp_3d`) over the collision-free
       3D voxel envelope under the 4 multi-directional 6-DOF arm wrenches.
     - 3D Ground-Structure Space-Truss Optimization (`solve_ground_truss_3d`) over collision-free 3D
       candidate struts between all connection bosses and interior Steiner hub nodes.
  4. Saves the 3D density fields (`arm-topopt-fields.npz`) and optimization telemetry (`arm-topopt-report.json`)
     to `artifacts/robot/body/arm/topopt/`.
"""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_k] = "1"

from pathlib import Path
import sys
import time
import numpy as np
import manifold3d as md
from scipy.ndimage import binary_dilation, gaussian_filter, label
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ARM_SHELL_DIR,
    ART,
    A_YAW,
    PRINTED,
    N_YAW,
    P_ELBOW,
    P_PITCH,
    P_ROLL,
    P_YAW,
    REPLACED_ARM_PARTS,
    RIGHT_ARM_SHELL_MOUNTS,
    ROOT,
    ROT_ROLL,
    STAGE_LINKS,
    STAGE_PART_BASENAMES,
    V_YAW,
    W_YAW,
    ensure_arm_shell_nut_assets,
    load_elasticity,
    orthotropic_matrix_MPa,
    read_json,
    stage_interface_specs,
    write_json,
)
from common import ARM_LOAD_WRENCHES, build_poses, fk, npz_to_mf

ELAST = load_elasticity()


def voxelize_manifold(mf: md.Manifold, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Fast vectorized Z-ray parity voxelizer for a watertight `manifold3d.Manifold`."""
    m = mf.to_mesh()
    verts = np.asarray(m.vert_properties[:, :3], dtype=np.float64)
    tris = np.asarray(m.tri_verts, dtype=np.int64)
    if len(tris) == 0:
        return np.zeros((len(xs), len(ys), len(zs)), dtype=bool)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    tmin_x = np.minimum(np.minimum(v0[:, 0], v1[:, 0]), v2[:, 0])
    tmax_x = np.maximum(np.maximum(v0[:, 0], v1[:, 0]), v2[:, 0])
    tmin_y = np.minimum(np.minimum(v0[:, 1], v1[:, 1]), v2[:, 1])
    tmax_y = np.maximum(np.maximum(v0[:, 1], v1[:, 1]), v2[:, 1])

    nx, ny, nz = len(xs), len(ys), len(zs)
    occ = np.zeros((nx, ny, nz), dtype=bool)
    xj = xs + 1e-4 * 0.37
    yj = ys + 1e-4 * 0.73
    for ix, x in enumerate(xj):
        cand_x = np.where((tmin_x <= x) & (tmax_x >= x))[0]
        if len(cand_x) == 0:
            continue
        for iy, y in enumerate(yj):
            idx = cand_x[(tmin_y[cand_x] <= y) & (tmax_y[cand_x] >= y)]
            if len(idx) == 0:
                continue
            a = v0[idx]
            b = v1[idx]
            c = v2[idx]
            x0, y0 = a[:, 0], a[:, 1]
            x1, y1 = b[:, 0], b[:, 1]
            x2, y2 = c[:, 0], c[:, 1]
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            valid = np.abs(den) > 1e-12
            if not np.any(valid):
                continue
            w0 = ((y1[valid] - y2[valid]) * (x - x2[valid]) + (x2[valid] - x1[valid]) * (y - y2[valid])) / den[valid]
            w1 = ((y2[valid] - y0[valid]) * (x - x2[valid]) + (x0[valid] - x2[valid]) * (y - y2[valid])) / den[valid]
            w2 = 1.0 - w0 - w1
            hit = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            if not np.any(hit):
                continue
            z_hits = w0[hit] * a[valid][hit, 2] + w1[hit] * b[valid][hit, 2] + w2[hit] * c[valid][hit, 2]
            counts = np.sum(z_hits[:, None] < zs[None, :], axis=0)
            occ[ix, iy, :] = (counts % 2) == 1
    return occ


def build_hex8_element_stiffness(dx: float, dy: float, dz: float, D: np.ndarray) -> np.ndarray:
    """Integrate 24x24 8-node hexahedral element stiffness matrix via 2x2x2 Gauss quadrature."""
    nodes = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    gp = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    Ke = np.zeros((24, 24), dtype=float)
    detJ = (dx * dy * dz) / 8.0
    invJ = np.diag([2.0 / dx, 2.0 / dy, 2.0 / dz])
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN_dxi = 0.125 * np.array(
                    [
                        [
                            nodes[a, 0] * (1 + nodes[a, 1] * eta) * (1 + nodes[a, 2] * zeta),
                            nodes[a, 1] * (1 + nodes[a, 0] * xi) * (1 + nodes[a, 2] * zeta),
                            nodes[a, 2] * (1 + nodes[a, 0] * xi) * (1 + nodes[a, 1] * eta),
                        ]
                        for a in range(8)
                    ]
                )
                dN_dx = dN_dxi @ invJ
                B = np.zeros((6, 24), dtype=float)
                for a in range(8):
                    c = 3 * a
                    gx, gy, gz = dN_dx[a]
                    B[0, c] = gx
                    B[1, c + 1] = gy
                    B[2, c + 2] = gz
                    B[3, c] = gy
                    B[3, c + 1] = gx
                    B[4, c + 1] = gz
                    B[4, c + 2] = gy
                    B[5, c] = gz
                    B[5, c + 2] = gx
                Ke += (B.T @ D @ B) * detJ
    return Ke


def build_blank_state_stage_domain(stage_key: str, env_parts: list, poses: list, specs: dict) -> dict:
    """Compute the blank-state 3D collision-free voxel domain, active connection bosses, and driver ray voids."""
    slink = f"R_{STAGE_LINKS[stage_key]}"
    stage_grids = {
        "stage1_shoulder_pitch_to_roll": ([69.5, -15.5, 320.5], [139.5, 24.5, 364.5], (36, 22, 24), 1),
        "stage2_shoulder_roll_to_yaw": ([101.0, -19.0, 286.5], [148.0, 26.5, 370.0], (26, 25, 36), 1),
        "stage3_upper_arm_yaw_to_elbow": ([126.0, -20.5, 241.5], [167.0, 7.5, 290.5], (24, 18, 28), 0),
        "stage4_forearm_elbow_to_wrist": ([124.0, -23.5, 170.5], [163.5, 17.5, 246.5], (22, 22, 34), 0),
    }
    bmin, bmax, (nx, ny, nz), layer_axis = stage_grids[stage_key]
    xs = np.linspace(bmin[0], bmax[0], nx)
    ys = np.linspace(bmin[1], bmax[1], ny)
    zs = np.linspace(bmin[2], bmax[2], nz)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    dz = float(zs[1] - zs[0])

    # Collect all 42-pose swept obstacles intersecting the bounding box
    src = read_json(ARM_SHELL_DIR / "source.json")
    s_bb = np.array([bmin, bmax], dtype=float)
    seen = set()
    obstacles = []
    for _, q in poses:
        T = fk(src["joints"], q)
        inv = np.linalg.inv(T[slink])
        for oname, olink, _, omf in env_parts:
            rel_M = inv @ T[olink]
            key = (oname, tuple(np.round(rel_M[:3].ravel(), 5)))
            if key in seen:
                continue
            seen.add(key)
            moved = omf.transform(rel_M[:3])
            o_bb = np.asarray(moved.bounding_box()).reshape(2, 3)
            if not np.all(np.minimum(s_bb[1], o_bb[1]) - np.maximum(s_bb[0], o_bb[0]) >= 0.0):
                continue
            obstacles.append(moved)

    obs_union = md.Manifold.batch_boolean(obstacles, md.OpType.Add)
    occ_obs = voxelize_manifold(obs_union, xs, ys, zs)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts_grid = np.stack([X, Y, Z], axis=-1)

    # Seed all active connection points (both support & load dual-cheek sleeves + integrated shell mounts)
    st_spec = specs[stage_key]
    passive_solid = np.zeros((nx, ny, nz), dtype=bool)
    passive_void = occ_obs.copy()
    sup_mask = np.zeros((nx, ny, nz), dtype=bool)
    load_mask = np.zeros((nx, ny, nz), dtype=bool)
    shell_mount_mask = np.zeros((nx, ny, nz), dtype=bool)

    connection_nodes = []

    for role_key in ("support", "load"):
        grp = st_spec[role_key]
        norms = grp.get("normals", [grp["normal"]] * len(grp["centres"]))
        ro = float(grp["sleeve_ro"])
        ri = float(grp["sleeve_ri"])
        h = float(grp["sleeve_len"])
        for idx, (pt, d) in enumerate(zip(grp["centres"], norms)):
            rel_v = pts_grid - pt[None, None, None, :]
            proj = np.einsum("ijkl,l->ijk", rel_v, d)
            perp = np.linalg.norm(rel_v - proj[..., None] * d[None, None, None, :], axis=-1)
            boss_ring = (proj >= -0.8) & (proj <= h + 1.2) & (perp >= ri * 0.7) & (perp <= ro + 2.8)
            bore_void = (proj >= -1.5) & (proj <= h + 26.0) & (perp < ri * 0.85)
            passive_solid |= boss_ring
            passive_void |= bore_void
            if role_key == "support":
                sup_mask |= boss_ring
            else:
                load_mask |= boss_ring
            connection_nodes.append({
                "id": f"{grp['name']}_{idx + 1}",
                "role": role_key,
                "center_mm": np.round(pt + 0.5 * h * d, 3).tolist(),
                "normal": np.round(d, 4).tolist(),
            })
        if "rocker_center" in grp:
            rc = grp["rocker_center"]
            rr = float(grp["rocker_r"])
            d0 = grp["normal"]
            rel_r = pts_grid - rc[None, None, None, :]
            proj_r = np.einsum("ijkl,l->ijk", rel_r, d0)
            perp_r = np.linalg.norm(rel_r - proj_r[..., None] * d0[None, None, None, :], axis=-1)
            passive_void |= (proj_r >= -2.0) & (proj_r <= h + 10.0) & (perp_r < rr)

    # Also seed the integrated hard-shell mounting posts on Stage 1 and Stage 4
    for m in RIGHT_ARM_SHELL_MOUNTS:
        if m["stage"] != stage_key:
            continue
        cx, cy, cz = m["center"]
        ax = np.asarray(m["axis"], dtype=float)
        pt_m = np.array([cx, cy, cz], dtype=float)
        rel_m = pts_grid - pt_m[None, None, None, :]
        proj_m = np.einsum("ijkl,l->ijk", rel_m, ax)
        perp_m = np.linalg.norm(rel_m - proj_m[..., None] * ax[None, None, None, :], axis=-1)
        post_ring = (np.abs(proj_m) <= 3.6) & (perp_m >= 1.5) & (perp_m <= 4.4)
        passive_solid |= post_ring
        shell_mount_mask |= post_ring
        connection_nodes.append({
            "id": m["id"],
            "role": "shell_mount",
            "center_mm": [cx, cy, cz],
            "normal": list(m["axis"]),
        })

    # Blank-state valid domain: collision-free interior connected to the connection bosses (plus passive_solid bosses)
    free_vox = (~passive_void) | passive_solid
    # Restrict exterior outside the hard-shell perimeter while giving the solver the full 3D interior cavity
    if stage_key == "stage1_shoulder_pitch_to_roll":
        shell_interior = (X >= 70.0) & (X <= 139.2) & (Y >= -15.0) & (Y <= 23.8) & (Z >= 321.0) & (Z <= 363.8)
    elif stage_key == "stage2_shoulder_roll_to_yaw":
        shell_interior = (X >= 101.5) & (X <= 147.2) & (Y >= -18.8) & (Y <= 25.8) & (Z >= 287.0) & (Z <= 369.5)
    elif stage_key == "stage3_upper_arm_yaw_to_elbow":
        shell_interior = (X >= 126.5) & (X <= 166.8) & (Y >= -20.0) & (Y <= 7.0) & (Z >= 242.0) & (Z <= 290.0)
    else:
        shell_interior = (X >= 124.2) & (X <= 163.2) & (Y >= -23.0) & (Y <= 17.0) & (Z >= 171.0) & (Z <= 246.0)

    valid_domain = (free_vox & shell_interior) | passive_solid
    # Ensure full 3D connectivity across the blank-state domain between support and load bosses
    labeled, n_comp = label(valid_domain)
    if n_comp > 1:
        seed_labels = np.unique(labeled[passive_solid])
        seed_labels = seed_labels[seed_labels > 0]
        valid_domain = np.isin(labeled, seed_labels) | passive_solid

    return {
        "stage_key": stage_key,
        "bmin": bmin,
        "bmax": bmax,
        "shape": (nx, ny, nz),
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "layer_axis": layer_axis,
        "valid_domain": valid_domain,
        "passive_solid": passive_solid & (~occ_obs),
        "passive_void": occ_obs,
        "sup_mask": sup_mask,
        "load_mask": load_mask,
        "shell_mount_mask": shell_mount_mask,
        "connection_nodes": connection_nodes,
        "obs_union": obs_union,
    }


def solve_simp_3d(dom: dict, target_vol_frac: float = 0.28, n_iters: int = 26, penal: float = 3.0) -> dict:
    """Run multi-load 3D orthotropic SIMP topology optimization over the blank-state collision-free voxel domain."""
    nx, ny, nz = dom["shape"]
    dx, dy, dz = dom["dx"], dom["dy"], dom["dz"]
    valid = dom["valid_domain"]
    p_solid = dom["passive_solid"]
    sup_el = dom["sup_mask"] & valid
    load_el = dom["load_mask"] & valid
    shell_el = dom["shell_mount_mask"] & valid

    el_idx = np.flatnonzero(valid.ravel())
    n_el = len(el_idx)

    # Map active hex8 elements to nodes on (nx+1, ny+1, nz+1) grid
    ix, iy, iz = np.unravel_index(el_idx, (nx, ny, nz))
    n_stride_x = (ny + 1) * (nz + 1)
    n_stride_y = nz + 1

    def _nid(x, y, z):
        return x * n_stride_x + y * n_stride_y + z

    el_nodes = np.column_stack(
        [
            _nid(ix, iy, iz),
            _nid(ix + 1, iy, iz),
            _nid(ix + 1, iy + 1, iz),
            _nid(ix, iy + 1, iz),
            _nid(ix, iy, iz + 1),
            _nid(ix + 1, iy, iz + 1),
            _nid(ix + 1, iy + 1, iz + 1),
            _nid(ix, iy + 1, iz + 1),
        ]
    )
    uniq_nodes, inv_nodes = np.unique(el_nodes, return_inverse=True)
    el_nodes_compact = inv_nodes.reshape(n_el, 8)
    n_nodes = len(uniq_nodes)
    n_dofs = 3 * n_nodes

    # Node coordinates
    ux, uy, uz = np.unravel_index(uniq_nodes, (nx + 1, ny + 1, nz + 1))
    xs_n = dom["bmin"][0] + ux * dx
    ys_n = dom["bmin"][1] + uy * dy
    zs_n = dom["bmin"][2] + uz * dz
    node_xyz = np.column_stack([xs_n, ys_n, zs_n])

    # Orthotropic Hooke matrix of the printed material (the definition's PRINTED_MATERIAL, robot/body/checks/data/materials.json)
    m = PRINTED
    E_vec = [m["E_inplane_MPa"], m["E_inplane_MPa"], m["E_inplane_MPa"]]
    E_vec[dom["layer_axis"]] = m["E_layer_MPa"]
    D = orthotropic_matrix_MPa(E_vec[0], E_vec[1], E_vec[2], nu=m["nu"], G=m["G_MPa"])
    Ke = build_hex8_element_stiffness(dx, dy, dz, D)

    # Precompute DOF indices per element
    el_dofs = np.zeros((n_el, 24), dtype=np.int32)
    for a in range(8):
        el_dofs[:, 3 * a : 3 * a + 3] = 3 * el_nodes_compact[:, a, None] + np.arange(3)[None, :]

    # Support & load node sets
    sup_el_local = np.flatnonzero(sup_el.ravel()[el_idx])
    load_el_local = np.flatnonzero(load_el.ravel()[el_idx])
    shell_el_local = np.flatnonzero(shell_el.ravel()[el_idx])

    sup_nodes = np.unique(el_nodes_compact[sup_el_local])
    load_nodes = np.unique(el_nodes_compact[load_el_local])
    shell_nodes = np.unique(el_nodes_compact[shell_el_local]) if len(shell_el_local) > 0 else np.array([], dtype=int)

    fixed_dofs = (3 * sup_nodes[:, None] + np.arange(3)[None, :]).ravel()
    free_mask = np.ones(n_dofs, dtype=bool)
    free_mask[fixed_dofs] = False
    free_dofs = np.flatnonzero(free_mask)

    # Build multi-load RHS vectors for the 4 6-DOF arm wrenches + shell-mount secondary stiffness load
    w_load_nodes = np.zeros(n_nodes, dtype=float)
    w_load_nodes[load_nodes] = 1.0
    origin = node_xyz[load_nodes].mean(axis=0)

    rhs_list = []
    for _, wrench_vec in ARM_LOAD_WRENCHES[dom["stage_key"]].items():
        f_nodal, _ = ELAST.minimum_weight_wrench(node_xyz, w_load_nodes, np.asarray(wrench_vec, dtype=float), origin)
        rhs_list.append(f_nodal.ravel())
    if len(shell_nodes) > 0:
        f_sh = np.zeros((n_nodes, 3), dtype=float)
        f_sh[shell_nodes] = np.array([12.0, -12.0, 15.0]) / len(shell_nodes)
        rhs_list.append(f_sh.ravel())
    RHS = np.column_stack(rhs_list)
    n_loads = RHS.shape[1]

    # Sparse assembly pattern
    ii = np.repeat(el_dofs[:, :, None], 24, axis=2).ravel()
    jj = np.repeat(el_dofs[:, None, :], 24, axis=1).ravel()
    ke_flat = Ke.ravel()

    # Initialize density rho
    rho = np.full(n_el, target_vol_frac, dtype=float)
    p_solid_local = p_solid.ravel()[el_idx]
    rho[p_solid_local] = 1.0
    rho_min = 1e-3

    history = []
    U = np.zeros((n_dofs, n_loads), dtype=float)

    for it in range(n_iters):
        stiff_scale = rho_min + (1.0 - rho_min) * (rho ** penal)
        vals = (stiff_scale[:, None] * ke_flat[None, :]).ravel()
        K = coo_matrix((vals, (ii, jj)), shape=(n_dofs, n_dofs)).tocsc()
        K_ff = K[free_dofs][:, free_dofs]
        diag_k = np.maximum(K_ff.diagonal(), 1e-6)
        M_prec = coo_matrix((1.0 / diag_k, (np.arange(len(free_dofs)), np.arange(len(free_dofs)))), shape=K_ff.shape).tocsc()

        total_comp = 0.0
        ce_total = np.zeros(n_el, dtype=float)
        for l_idx in range(n_loads):
            u_f, _ = cg(K_ff, RHS[free_dofs, l_idx], x0=U[free_dofs, l_idx], M=M_prec, rtol=1e-4, maxiter=220)
            U[free_dofs, l_idx] = u_f
            u_e = U[:, l_idx][el_dofs]
            ce_l = np.einsum("ei,ij,ej->e", u_e, Ke, u_e)
            w_l = 1.0 if l_idx < 4 else 0.35
            total_comp += w_l * float(np.dot(RHS[free_dofs, l_idx], u_f))
            ce_total += w_l * ce_l

        dc = -penal * (1.0 - rho_min) * (rho ** (penal - 1.0)) * ce_total
        # 3D Gaussian sensitivity filter on the voxel grid
        dc_grid = np.zeros((nx, ny, nz), dtype=float)
        rho_grid = np.zeros((nx, ny, nz), dtype=float)
        dc_grid.ravel()[el_idx] = rho * (-dc)
        rho_grid.ravel()[el_idx] = rho
        dc_filt = gaussian_filter(dc_grid, sigma=1.15) / np.maximum(gaussian_filter(rho_grid, sigma=1.15), 1e-6)
        sens = np.maximum(dc_filt.ravel()[el_idx], 1e-12)

        # Optimality criteria (OC) update
        l1, l2 = 1e-9, 1e9
        move = 0.18
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            rho_new = np.clip(
                rho * np.sqrt(sens / lmid),
                np.maximum(0.02, rho - move),
                np.minimum(1.0, rho + move),
            )
            rho_new[p_solid_local] = 1.0
            if rho_new.mean() > target_vol_frac:
                l1 = lmid
            else:
                l2 = lmid
        rho = rho_new
        history.append({"iter": it + 1, "compliance": round(total_comp, 3), "mean_rho": round(float(rho.mean()), 4)})

    rho_3d = np.zeros((nx, ny, nz), dtype=np.float32)
    rho_3d.ravel()[el_idx] = rho.astype(np.float32)
    strain_3d = np.zeros((nx, ny, nz), dtype=np.float32)
    strain_3d.ravel()[el_idx] = (ce_total * (rho ** penal)).astype(np.float32)

    return {
        "rho_3d": rho_3d,
        "strain_3d": strain_3d,
        "history": history,
        "initial_compliance": history[0]["compliance"],
        "final_compliance": history[-1]["compliance"],
        "compliance_reduction_pct": round((1.0 - history[-1]["compliance"] / max(history[0]["compliance"], 1e-9)) * 100.0, 2),
        "active_voxels": int(n_el),
    }


def solve_ground_truss_3d(dom: dict) -> dict:
    """Discover optimal 3D space-truss struts connecting all support, load, and shell-mount connection points
    across the blank-state collision-free 3D flexion domain."""
    nodes_info = dom["connection_nodes"]
    valid = dom["valid_domain"]
    xs, ys, zs = dom["xs"], dom["ys"], dom["zs"]

    # Sample interior high-density Steiner hub points inside the collision-free 3D envelope
    pts = [np.asarray(n["center_mm"], dtype=float) for n in nodes_info]
    labels = [n["id"] for n in nodes_info]
    roles = [n["role"] for n in nodes_info]

    # Add 4 interior 3D Steiner hub nodes at quartiles of the collision-free domain
    free_ijk = np.argwhere(valid)
    if len(free_ijk) > 0:
        for q in (0.25, 0.45, 0.65, 0.82):
            idx_q = free_ijk[int(q * (len(free_ijk) - 1))]
            p_st = np.array([xs[idx_q[0]], ys[idx_q[1]], zs[idx_q[2]]], dtype=float)
            pts.append(p_st)
            labels.append(f"steiner_hub_{int(q*100)}")
            roles.append("steiner")

    pts_arr = np.array(pts, dtype=float)
    n_n = len(pts_arr)
    struts = []
    for i in range(n_n):
        for j in range(i + 1, n_n):
            p0, p1 = pts_arr[i], pts_arr[j]
            length = float(np.linalg.norm(p1 - p0))
            if length < 5.0 or length > 78.0:
                continue
            # Check straight-line collision clearance across the blank-state 3D domain
            ts = np.linspace(0.15, 0.85, 12)
            seg = p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]
            ix = np.clip(np.searchsorted(xs, seg[:, 0]), 0, len(xs) - 1)
            iy = np.clip(np.searchsorted(ys, seg[:, 1]), 0, len(ys) - 1)
            iz = np.clip(np.searchsorted(zs, seg[:, 2]), 0, len(zs) - 1)
            clear_frac = float(np.mean(valid[ix, iy, iz]))
            if clear_frac >= 0.65:
                struts.append({
                    "from": labels[i],
                    "to": labels[j],
                    "from_role": roles[i],
                    "to_role": roles[j],
                    "p0_mm": np.round(p0, 2).tolist(),
                    "p1_mm": np.round(p1, 2).tolist(),
                    "length_mm": round(length, 2),
                    "clearance_fraction": round(clear_frac, 3),
                })
    return {
        "nodes_count": n_n,
        "candidate_collision_free_struts_count": len(struts),
        "top_struts": struts[:24],
    }


def main():
    t0 = time.time()
    ensure_arm_shell_nut_assets()
    mp = read_json(ARM_SHELL_DIR / "manufactured-parts.json")
    sp = read_json(ARM_SHELL_DIR / "source-packages.json")
    replaced = set(mp["replaced_source"]) | REPLACED_ARM_PARTS

    env_parts = []
    for p in sp:
        pname = p["name"]
        if pname.startswith("upper__R_") and p.get("exact") and pname not in replaced:
            env_parts.append((pname, p["owner"], "source", npz_to_mf(ROOT / p["exact"]["path"])))
    for p in mp["parts"]:
        name = p["name"]
        if name.startswith("R_") and name not in replaced and "insert" not in name:
            env_parts.append((name, p["owning_link"], p["role"], npz_to_mf(ROOT / p["lossless_mesh"]["path"])))

    poses = build_poses()
    specs = stage_interface_specs()

    out_dir = ART / "topopt"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dict = {}
    report = {
        "method": "Blank-State 3D Multi-Load Orthotropic SIMP + 3D Ground-Structure Space-Truss Discovery",
        "sheet_metal_seed_used": False,
        "poses_enforced": len(poses),
        "total_active_connections_per_arm": 46,
        "total_steel_sleeves_per_arm": 38,
        "total_integrated_shell_mounts_per_arm": 8,
        "stages": {},
    }

    for stage_key in STAGE_PART_BASENAMES:
        ts = time.time()
        dom = build_blank_state_stage_domain(stage_key, env_parts, poses, specs)
        simp = solve_simp_3d(dom, target_vol_frac=0.28, n_iters=26)
        truss = solve_ground_truss_3d(dom)

        npz_dict[f"{stage_key}__rho"] = simp["rho_3d"]
        npz_dict[f"{stage_key}__strain"] = simp["strain_3d"]
        npz_dict[f"{stage_key}__valid"] = dom["valid_domain"].astype(np.uint8)
        npz_dict[f"{stage_key}__xs"] = dom["xs"].astype(np.float32)
        npz_dict[f"{stage_key}__ys"] = dom["ys"].astype(np.float32)
        npz_dict[f"{stage_key}__zs"] = dom["zs"].astype(np.float32)

        report["stages"][stage_key] = {
            "grid_shape": list(dom["shape"]),
            "voxel_size_mm": [round(dom["dx"], 2), round(dom["dy"], 2), round(dom["dz"], 2)],
            "active_collision_free_voxels": simp["active_voxels"],
            "connection_nodes_count": len(dom["connection_nodes"]),
            "connection_nodes": dom["connection_nodes"],
            "simp_initial_compliance": simp["initial_compliance"],
            "simp_final_compliance": simp["final_compliance"],
            "simp_compliance_reduction_pct": simp["compliance_reduction_pct"],
            "simp_history": simp["history"],
            "ground_truss_summary": truss,
            "elapsed_s": round(time.time() - ts, 2),
        }
        print(
            f"[{stage_key}] vox={simp['active_voxels']} | conns={len(dom['connection_nodes'])} | "
            f"comp: {simp['initial_compliance']:.1f} -> {simp['final_compliance']:.1f} "
            f"(-{simp['compliance_reduction_pct']:.1f}%) | struts={truss['candidate_collision_free_struts_count']} | "
            f"dt={time.time()-ts:.2f}s"
        )

    np.savez_compressed(out_dir / "arm-topopt-fields.npz", **npz_dict)
    write_json(out_dir / "arm-topopt-report.json", report)
    print(f"Saved blank-state 3D SIMP + ground-truss results in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
