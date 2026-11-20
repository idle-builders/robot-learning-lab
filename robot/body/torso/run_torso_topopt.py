"""Step 2 — Blank-State 3D Multi-Load SIMP Topology Optimization for the 4 Center-Body Torso Carrier Stages."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = ROOT / "robot/body/torso"
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import common


def element_stiffness_3d(E: float = 1.0, nu: float = 0.35) -> np.ndarray:
    """8-node hexahedral element stiffness matrix for 3D linear elasticity."""
    gp = 1.0 / np.sqrt(3.0)
    gauss = [(-gp, -gp, -gp), (gp, -gp, -gp), (gp, gp, -gp), (-gp, gp, -gp),
             (-gp, -gp, gp), (gp, -gp, gp), (gp, gp, gp), (-gp, gp, gp)]
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = c * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
    ], dtype=float)

    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)

    KE = np.zeros((24, 24), dtype=float)
    for xi, eta, zeta in gauss:
        dN = 0.125 * np.column_stack([
            signs[:, 0] * (1 + signs[:, 1] * eta) * (1 + signs[:, 2] * zeta),
            signs[:, 1] * (1 + signs[:, 0] * xi) * (1 + signs[:, 2] * zeta),
            signs[:, 2] * (1 + signs[:, 0] * xi) * (1 + signs[:, 1] * eta),
        ])
        B = np.zeros((6, 24), dtype=float)
        for a in range(8):
            c0 = 3 * a
            nx, ny, nz = dN[a]
            B[0, c0] = nx
            B[1, c0 + 1] = ny
            B[2, c0 + 2] = nz
            B[3, c0] = ny
            B[3, c0 + 1] = nx
            B[4, c0 + 1] = nz
            B[4, c0 + 2] = ny
            B[5, c0] = nz
            B[5, c0 + 2] = nx
        KE += B.T @ D @ B
    return KE


def run_stage_simp_3d(stage: str, nx: int = 18, ny: int = 10, nz: int = 16, volfrac: float = 0.44, n_iter: int = 24) -> dict:
    """Solve multi-load 3D SIMP topology optimization on the stage design domain."""
    t0 = time.monotonic()
    spec = common.stage_interface_specs()[stage]
    base_shape = common.load_baseline_stage(stage)
    bb = base_shape.bounding_box()
    bmin = np.array([bb.min.X, bb.min.Y, bb.min.Z], dtype=float)
    bmax = np.array([bb.max.X, bb.max.Y, bb.max.Z], dtype=float)
    span = np.maximum(bmax - bmin, 4.0)

    xs = np.linspace(bmin[0] + 0.5 * span[0] / nx, bmax[0] - 0.5 * span[0] / nx, nx)
    ys = np.linspace(bmin[1] + 0.5 * span[1] / ny, bmax[1] - 0.5 * span[1] / ny, ny)
    zs = np.linspace(bmin[2] + 0.5 * span[2] / nz, bmax[2] - 0.5 * span[2] / nz, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([X, Y, Z], axis=-1)

    passive_solid = np.zeros((nx, ny, nz), dtype=bool)
    support_nodes_mask = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    load_elem_indices = []

    n_xs = np.linspace(bmin[0], bmax[0], nx + 1)
    n_ys = np.linspace(bmin[1], bmax[1], ny + 1)
    n_zs = np.linspace(bmin[2], bmax[2], nz + 1)
    NX, NY, NZ = np.meshgrid(n_xs, n_ys, n_zs, indexing="ij")
    node_coords = np.stack([NX, NY, NZ], axis=-1)

    r_scale = 0.09 * float(np.linalg.norm(span))
    for bore in spec["support_bores"]:
        c = np.array(bore["center"], dtype=float)
        d_elem = np.linalg.norm(centers - c, axis=-1)
        passive_solid |= d_elem <= max(bore["r_od"] * 1.6, r_scale * 0.45)
        d_node = np.linalg.norm(node_coords - c, axis=-1)
        support_nodes_mask |= d_node <= max(bore["r_od"] * 2.2, r_scale * 0.72)

    if not np.any(support_nodes_mask):
        c0 = np.array(spec["support_bores"][0]["center"], dtype=float)
        d_node = np.linalg.norm(node_coords - c0, axis=-1)
        support_nodes_mask[d_node <= np.percentile(d_node, 8.0)] = True

    for bore in spec["load_bores"]:
        c = np.array(bore["center"], dtype=float)
        d_elem = np.linalg.norm(centers - c, axis=-1)
        mask = d_elem <= max(bore["r_od"] * 1.6, r_scale * 0.45)
        passive_solid |= mask
        idx = int(np.argmin(d_elem))
        load_elem_indices.append(np.unravel_index(idx, (nx, ny, nz)))

    KE = element_stiffness_3d()
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    n_dofs = 3 * n_nodes

    def node_id(ix: int, iy: int, iz: int) -> int:
        return ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz

    n_elem = nx * ny * nz
    edof_mat = np.zeros((n_elem, 24), dtype=int)
    eid = 0
    offsets = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for a, (dx, dy, dz) in enumerate(offsets):
                    nid = node_id(ix + dx, iy + dy, iz + dz)
                    edof_mat[eid, 3 * a:3 * a + 3] = [3 * nid, 3 * nid + 1, 3 * nid + 2]
                eid += 1

    iK = np.kron(edof_mat, np.ones((24, 1), dtype=int)).ravel()
    jK = np.kron(edof_mat, np.ones((1, 24), dtype=int)).ravel()

    fixed_nids = np.flatnonzero(support_nodes_mask.ravel())
    fixed_dofs = np.concatenate([3 * fixed_nids, 3 * fixed_nids + 1, 3 * fixed_nids + 2])
    free_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)

    # 3 orthogonal multi-limb load cases:
    # Case 0: Vertical gravity/impact + shoulder vertical reaction (-Z)
    # Case 1: Bilateral shoulder pitch + neck sagittal moment (Y-Z shear + pitch couple)
    # Case 2: Bilateral hip yaw + shoulder lateral torsion (X-Y torsion couple)
    F = np.zeros((n_dofs, 3), dtype=float)
    for k, (ix, iy, iz) in enumerate(load_elem_indices):
        nid = node_id(min(ix, nx), min(iy, ny), min(iz, nz))
        sign = 1.0 if (k % 2 == 0) else -1.0
        F[3 * nid + 2, 0] -= 45.0
        F[3 * nid + 1, 1] += 32.0 * sign
        F[3 * nid + 2, 1] -= 22.0
        F[3 * nid + 0, 2] += 38.0 * sign
        F[3 * nid + 1, 2] -= 18.0 * sign

    x = np.full((nx, ny, nz), volfrac, dtype=float)
    x[passive_solid] = 1.0
    E0 = 1.0
    Emin = 1e-4
    penal = 3.0
    history = []

    for it in range(n_iter):
        x_flat = x.ravel()
        ce_total = np.zeros(n_elem, dtype=float)
        obj_total = 0.0
        sK = ((Emin + (x_flat ** penal) * (E0 - Emin))[:, None] * KE.ravel()[None, :]).ravel()
        K = coo_matrix((sK, (iK, jK)), shape=(n_dofs, n_dofs)).tocsc()
        K_ff = K[free_dofs, :][:, free_dofs]
        U = np.zeros((n_dofs, 3), dtype=float)
        U[free_dofs, :] = spsolve(K_ff, F[free_dofs, :])

        for lc in range(3):
            Ue = U[edof_mat, lc]
            ce = np.einsum("ij,jk,ik->i", Ue, KE, Ue)
            ce_total += ce
            obj_total += float(np.sum((Emin + (x_flat ** penal) * (E0 - Emin)) * ce))

        dc = -penal * (E0 - Emin) * (x_flat ** (penal - 1.0)) * ce_total
        dc_grid = gaussian_filter(dc.reshape((nx, ny, nz)), sigma=0.85)
        dc_flat = np.minimum(dc_grid.ravel(), -1e-12)

        l1, l2, move = 0.0, 1e6, 0.16
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            x_new = np.maximum(
                0.08,
                np.maximum(
                    x_flat - move,
                    np.minimum(1.0, np.minimum(x_flat + move, x_flat * np.sqrt(-dc_flat / lmid))),
                ),
            )
            x_new[passive_solid.ravel()] = 1.0
            if np.mean(x_new) > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.max(np.abs(x_new - x_flat)))
        x = x_new.reshape((nx, ny, nz))
        history.append({
            "iteration": it + 1,
            "compliance_objective": round(obj_total, 4),
            "volume_fraction": round(float(np.mean(x)), 4),
            "max_density_change": round(change, 5),
        })

    vm_proxy = (x.ravel() ** 1.5) * np.sqrt(np.maximum(ce_total, 0.0))
    vm_grid = vm_proxy.reshape((nx, ny, nz))

    return {
        "stage": stage,
        "grid_shape": [nx, ny, nz],
        "bounds_mm": [bmin.round(3).tolist(), bmax.round(3).tolist()],
        "target_volfrac": volfrac,
        "final_volfrac": round(float(np.mean(x)), 4),
        "initial_compliance": history[0]["compliance_objective"],
        "final_compliance": history[-1]["compliance_objective"],
        "compliance_reduction_pct": round(100.0 * (1.0 - history[-1]["compliance_objective"] / max(history[0]["compliance_objective"], 1e-9)), 2),
        "elapsed_s": round(time.monotonic() - t0, 3),
        "history": history,
        "density_grid": x,
        "stress_grid": vm_grid,
        "xs": xs,
        "ys": ys,
        "zs": zs,
    }


def main() -> None:
    common.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    npz_payload = {}
    report_stages = {}

    configs = {
        "stage1_center_torso_spine_chassis": {"nx": 20, "ny": 10, "nz": 22, "volfrac": 0.44},
        "stage2_lower_pelvis_battery_cradle": {"nx": 14, "ny": 14, "nz": 14, "volfrac": 0.46},
        "stage3_front_power_electronics_carrier": {"nx": 16, "ny": 10, "nz": 18, "volfrac": 0.45},
        "stage4_rear_imu_adc_carrier": {"nx": 12, "ny": 8, "nz": 16, "volfrac": 0.48},
    }

    for stage in common.STAGES:
        cfg = configs[stage]
        res = run_stage_simp_3d(stage, **cfg)
        npz_payload[f"{stage}_density"] = res["density_grid"].astype(np.float32)
        npz_payload[f"{stage}_stress"] = res["stress_grid"].astype(np.float32)
        npz_payload[f"{stage}_xs"] = res["xs"].astype(np.float32)
        npz_payload[f"{stage}_ys"] = res["ys"].astype(np.float32)
        npz_payload[f"{stage}_zs"] = res["zs"].astype(np.float32)

        report_stages[stage] = {k: v for k, v in res.items() if k not in ("density_grid", "stress_grid", "xs", "ys", "zs")}
        print(
            f"[SIMP 3D] {stage}: grid={res['grid_shape']} volfrac={res['final_volfrac']:.3f} "
            f"compliance {res['initial_compliance']:.2f} -> {res['final_compliance']:.2f} "
            f"(-{res['compliance_reduction_pct']:.1f}%) in {res['elapsed_s']:.2f}s"
        )

    npz_path = common.ARTIFACT_DIR / "torso-topopt-fields.npz"
    np.savez_compressed(npz_path, **npz_payload)
    report_path = common.ARTIFACT_DIR / "torso-topopt-report.json"
    report_path.write_text(json.dumps({
        "schema": "cove50.torso.simp_topopt_3d.v1",
        "study_id": "cove50-torso-fdm-carrier-20261105",
        "npz_file": str(npz_path.relative_to(ROOT)),
        "stages": report_stages,
    }, indent=2) + "\n")
    print(f"Saved {npz_path.relative_to(ROOT)} and {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
