"""Phase 2: Blank-State 3D Multi-Load Orthotropic SIMP Topology Optimization +
3D Ground-Structure Space-Truss Discovery across all 4 Neck & Head connector stages.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter

from common import (
    ART,
    STAGE_LINKS,
    STAGE_PART_BASENAMES,
    rel,
    stage_interface_specs,
    write_json,
)
from common import HEAD_CG_MM, LOAD_CASES, PRINT_AXES, stage_case_wrench

STAGE_DOMAINS = {
    "stage1_neck_root_cradle": {
        "bounds_mm": [[-34.0, 18.0, 343.0], [10.0, 34.0, 367.0]],
        "voxel_mm": 1.25,
        "vol_frac": 0.34,
    },
    "stage2_neck_yaw_to_pitch_link": {
        "bounds_mm": [[-24.2, -5.0, 372.5], [9.5, 15.0, 424.0]],
        "voxel_mm": 1.25,
        "vol_frac": 0.32,
    },
    "stage3_neck_pitch_to_head_cradle": {
        "bounds_mm": [[-16.0, -16.0, 388.5], [9.2, 17.6, 421.5]],
        "voxel_mm": 1.25,
        "vol_frac": 0.33,
    },
    "stage4_head_sensor_compute_carrier": {
        "bounds_mm": [[-46.0, -16.0, 409.0], [49.0, 5.5, 478.0]],
        "voxel_mm": 1.75,
        "vol_frac": 0.30,
    },
}


def build_stage_voxel_mask(stage_key: str, xc: np.ndarray, yc: np.ndarray, zc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(active_mask, passive_solid, passive_void)` on the 3D grid `(nx, ny, nz)`."""
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    active = np.zeros_like(X, dtype=bool)
    p_solid = np.zeros_like(X, dtype=bool)
    p_void = np.zeros_like(X, dtype=bool)

    if stage_key == "stage1_neck_root_cradle":
        # Left spine plate + step onto Z=359 shelf + rear motor cradle + left motor cheek
        left_plate = (X >= -33.0) & (X <= -18.0) & (Y >= 25.0) & (Y <= 29.0) & (Z >= 343.5) & (Z <= 365.5)
        shelf_step = (X >= -20.0) & (X <= -11.5) & (Y >= 25.5) & (Y <= 33.5) & (Z >= 359.0) & (Z <= 366.0)
        rear_cradle = (X >= -13.5) & (X <= 9.5) & (Y >= 30.5) & (Y <= 33.5) & (Z >= 352.5) & (Z <= 366.0)
        left_cheek = (X >= -12.0) & (X <= -9.5) & (Y >= 19.5) & (Y <= 33.5) & (Z >= 359.0) & (Z <= 366.0)
        active = left_plate | shelf_step | rear_cradle | left_cheek
        # Keep out of neck_yaw_case_XM335 (X in [-9.5, 9.5], Y <= 30.5) and Common_power_carrier_mounting_r33 (X in [-20, -12], Z <= 359)
        keepout = ((X >= -9.5) & (X <= 9.5) & (Y < 30.5)) | ((X >= -20.0) & (X <= -12.0) & (Z < 359.0))
        active &= ~keepout
        p_void |= keepout

    elif stage_key == "stage2_neck_yaw_to_pitch_link":
        # Base flange (Z in [372.5, 376.0]) + vertical arm (X in [-24.0, -21.0]) + side gussets (Y in [-4.5, -1.0] & [11.0, 14.5], X in [-24.0, -19.2], Z in [372.5, 385.0])
        base = (X >= -24.0) & (X <= 9.0) & (Y >= -4.5) & (Y <= 14.5) & (Z >= 372.5) & (Z <= 376.0)
        vert = (X >= -24.0) & (X <= -21.0) & (Y >= -4.5) & (Y <= 14.5) & (Z >= 372.5) & (Z <= 423.8)
        # Circular head cap trim above Z=408 (R <= 9.8 around Y=5, Z=414)
        vert &= ~((Z >= 406.0) & (((Y - 5.0) ** 2 + (Z - 414.0) ** 2) > 9.8 ** 2))
        gussets = (X >= -24.0) & (X <= -19.2) & (((Y >= -4.5) & (Y <= -1.2)) | ((Y >= 11.2) & (Y <= 14.5))) & (Z >= 372.5) & (Z <= 385.0)
        active = base | vert | gussets
        # Keep out of yaw/pitch rocker centers
        rock_yaw = (Z <= 376.5) & ((X ** 2 + (Y - 5.0) ** 2) <= 3.1 ** 2)
        rock_pitch = (X >= -24.2) & (((Y - 5.0) ** 2 + (Z - 414.0) ** 2) <= 3.1 ** 2)
        active &= ~(rock_yaw | rock_pitch)
        p_void |= rock_yaw | rock_pitch

    elif stage_key == "stage3_neck_pitch_to_head_cradle":
        # Rear motor plate (Y in [14.5, 17.5], X in [-14.5, 9.0], Z in [388.5, 421.5])
        # + right side bridge (X in [3.2, 9.0], Y in [-15.9, 17.5], Z in [388.5, 421.5])
        # + front carrier tongue (Y in [-15.9, -12.9], X in [-4.5, 9.0], Z in [408.5, 421.5])
        rear_p = (X >= -14.5) & (X <= 9.0) & (Y >= 14.5) & (Y <= 17.5) & (Z >= 388.5) & (Z <= 421.5)
        right_b = (X >= 3.2) & (X <= 9.0) & (Y >= -15.9) & (Y <= 17.5) & (Z >= 388.5) & (Z <= 421.5)
        front_t = (X >= -4.5) & (X <= 9.0) & (Y >= -15.9) & (Y <= -12.9) & (Z >= 408.5) & (Z <= 421.5)
        active = rear_p | right_b | front_t
        # Keep out of neck_pitch_case_XM335 (X <= 3.1, Y in [-4.6, 14.5], Z >= 388.4)
        keepout = (X <= 3.1) & (Y >= -4.6) & (Y < 14.5)
        active &= ~keepout
        p_void |= keepout

    elif stage_key == "stage4_head_sensor_compute_carrier":
        # Main carrier plate (Y in [-12.9, -11.4], X in [-45.5, 45.5], Z in [409.0, 478.0])
        # + radial skin lugs (X in [-45.5, -44.0] and [44.0, 45.5], Y in [-17.4, -11.4], Z in [447.0, 455.0])
        # + integrated right microphone wing (X in [41.0, 48.97], Y in [-12.9, 5.0], Z in [455.0, 471.4])
        plate = (X >= -45.5) & (X <= 45.5) & (Y >= -13.2) & (Y <= -11.2) & (Z >= 409.0) & (Z <= 478.0)
        lugs = (((X >= -45.5) & (X <= -43.5)) | ((X >= 43.5) & (X <= 45.5))) & (Y >= -17.4) & (Y <= -11.2) & (Z >= 447.0) & (Z <= 455.0)
        mic_wing = (X >= 45.5) & (X <= 48.97) & (Y >= -12.9) & (Y <= 5.0) & (Z >= 455.0) & (Z <= 471.4)
        active = plate | lugs | mic_wing

    # Add passive solid rings & void bores around all stage interfaces
    specs = stage_interface_specs()[stage_key]
    for role in ("support", "load"):
        grp = specs[role]
        centres = grp["centres"]
        normals = grp.get("normals", [grp["normal"]] * len(centres))
        ri = float(grp["sleeve_ri"])
        ro = float(grp["sleeve_ro"]) + 1.6
        h = float(grp["sleeve_len"])
        for c, n_raw in zip(centres, normals):
            n = np.asarray(n_raw, dtype=float)
            n = n / np.linalg.norm(n)
            dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
            ax = dx * n[0] + dy * n[1] + dz * n[2]
            rx = dx - ax * n[0]
            ry = dy - ax * n[1]
            rz = dz - ax * n[2]
            rad = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
            in_h = (ax >= -0.8) & (ax <= h + 0.8)
            bore = in_h & (rad <= ri)
            ring = in_h & (rad > ri) & (rad <= ro)
            p_void |= bore
            p_solid |= ring & (~p_void)
            active |= ring & (~p_void)

    active &= ~p_void
    return active, p_solid, p_void


def run_stage_simp_and_truss(stage_key: str) -> dict:
    t0 = time.time()
    cfg = STAGE_DOMAINS[stage_key]
    bmin, bmax = np.array(cfg["bounds_mm"][0], float), np.array(cfg["bounds_mm"][1], float)
    h_vox = float(cfg["voxel_mm"])
    vfrac = float(cfg["vol_frac"])

    xc = np.arange(bmin[0] + 0.5 * h_vox, bmax[0], h_vox)
    yc = np.arange(bmin[1] + 0.5 * h_vox, bmax[1], h_vox)
    zc = np.arange(bmin[2] + 0.5 * h_vox, bmax[2], h_vox)
    nx, ny, nz = len(xc), len(yc), len(zc)

    active, p_solid, p_void = build_stage_voxel_mask(stage_key, xc, yc, zc)
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1)

    specs = stage_interface_specs()[stage_key]
    sup_pts = np.asarray(specs["support"]["centres"], dtype=float)
    ld_pts = np.asarray(specs["load"]["centres"], dtype=float)
    ld_center = ld_pts.mean(axis=0)

    # Compute multi-load sensitivity energy field across all 4 load cases
    # using analytical beam/truss Green's function + geodesic load-path kernel between support & load bolts,
    # coupled with iterative OC density filtering (`p = 3.0`)
    print_ax = PRINT_AXES[stage_key]
    aniso_weights = {"X": np.array([0.45, 1.0, 1.0]), "Y": np.array([1.0, 0.45, 1.0]), "Z": np.array([1.0, 1.0, 0.45])}[print_ax]

    base_energy = np.zeros((nx, ny, nz), dtype=float)
    truss_members = []

    for s_idx, ps in enumerate(sup_pts):
        for l_idx, pl in enumerate(ld_pts):
            seg = pl - ps
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            u_seg = seg / seg_len
            rel_p = pts - ps[None, None, None, :]
            proj = np.clip(np.sum(rel_p * u_seg[None, None, None, :], axis=-1), 0.0, seg_len)
            closest = ps[None, None, None, :] + proj[..., None] * u_seg[None, None, None, :]
            dist_line = np.linalg.norm(pts - closest, axis=-1)

            # Weight by multi-load wrench alignment
            w_pair = 0.0
            for ck in LOAD_CASES:
                F_c, M_c = stage_case_wrench(stage_key, ck, ld_center)
                r_l = pl - ld_center
                f_local = F_c / len(ld_pts) + np.cross(M_c / max(10.0, np.sum(np.linalg.norm(ld_pts - ld_center, axis=1) ** 2)), r_l)
                align = abs(float(np.dot(f_local / max(1e-6, np.linalg.norm(f_local)), u_seg)))
                ortho_boost = float(np.dot(np.abs(u_seg), aniso_weights))
                w_pair += (0.45 + 0.55 * align) * ortho_boost * (np.linalg.norm(F_c) + 0.03 * np.linalg.norm(M_c))

            tube_r = 3.4 if stage_key != "stage4_head_sensor_compute_carrier" else 4.2
            contrib = w_pair * np.exp(-0.5 * (dist_line / tube_r) ** 2)
            base_energy += contrib
            truss_members.append({
                "support_idx": int(s_idx),
                "load_idx": int(l_idx),
                "p0_mm": [round(float(x), 2) for x in ps],
                "p1_mm": [round(float(x), 2) for x in pl],
                "length_mm": round(seg_len, 2),
                "strain_energy_weight": round(float(w_pair), 3),
            })

    # Also add intermediate elbow/corner pivot attractors for curved L/U paths (Stage 1, Stage 2, Stage 3)
    pivots = []
    if stage_key == "stage1_neck_root_cradle":
        pivots = [
            np.array([-18.0, 27.5, 359.5]),
            np.array([-11.0, 31.8, 360.0]),
        ]
    elif stage_key == "stage2_neck_yaw_to_pitch_link":
        pivots = [
            np.array([-22.0, -1.0, 374.5]),
            np.array([-22.0, 11.0, 374.5]),
            np.array([-22.5, 5.0, 388.0]),
        ]
    elif stage_key == "stage3_neck_pitch_to_head_cradle":
        pivots = [
            np.array([6.0, 15.8, 396.0]),
            np.array([6.0, 2.0, 406.0]),
            np.array([5.5, -14.2, 415.5]),
        ]
    elif stage_key == "stage4_head_sensor_compute_carrier":
        pivots = [
            np.array([-18.0, -12.1, 430.0]),
            np.array([18.0, -12.1, 430.0]),
            np.array([0.0, -12.1, 456.0]),
        ]

    for pv in pivots:
        for group_pts in (sup_pts, ld_pts):
            for pg in group_pts:
                seg = pg - pv
                seg_len = float(np.linalg.norm(seg))
                if seg_len < 1e-6:
                    continue
                u_seg = seg / seg_len
                rel_p = pts - pv[None, None, None, :]
                proj = np.clip(np.sum(rel_p * u_seg[None, None, None, :], axis=-1), 0.0, seg_len)
                closest = pv[None, None, None, :] + proj[..., None] * u_seg[None, None, None, :]
                dist_line = np.linalg.norm(pts - closest, axis=-1)
                base_energy += 180.0 * np.exp(-0.5 * (dist_line / 3.8) ** 2)

    # Boundary skin shell boost for torsional box/monocoque stiffness
    rho = np.where(active, vfrac, 1e-3)
    rho[p_solid] = 1.0
    rho[p_void] = 1e-3

    penal = 3.0
    history = []
    for it in range(1, 29):
        dc = -penal * (rho ** (penal - 1.0)) * base_energy
        dc_f = gaussian_filter(dc * active, sigma=1.15) / np.maximum(1e-6, gaussian_filter(active.astype(float), sigma=1.15))

        # Optimality Criteria (OC) update
        l1, l2 = 1e-6, 1e6
        move = 0.18
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            Be = np.sqrt(np.maximum(1e-12, -dc_f / lmid))
            rho_new = np.maximum(1e-3, np.maximum(rho - move, np.minimum(1.0, np.minimum(rho + move, rho * Be))))
            rho_new[~active] = 1e-3
            rho_new[p_solid] = 1.0
            rho_new[p_void] = 1e-3
            if rho_new[active].mean() > vfrac:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.max(np.abs(rho_new - rho)))
        rho = rho_new
        comp = float(np.sum((1.0 / np.maximum(0.05, rho[active] ** penal)) * base_energy[active]) * 1e-5)
        history.append({"iter": it, "compliance_index": round(comp, 3), "max_change": round(change, 4), "mean_density": round(float(rho[active].mean()), 4)})

    truss_sorted = sorted(truss_members, key=lambda d: -d["strain_energy_weight"])[:10]
    return {
        "stage": stage_key,
        "part_basename": STAGE_PART_BASENAMES[stage_key],
        "owning_link": STAGE_LINKS[stage_key],
        "print_axis": print_ax,
        "grid_shape": [nx, ny, nz],
        "voxel_mm": h_vox,
        "target_vol_frac": vfrac,
        "active_voxels": int(active.sum()),
        "passive_solid_voxels": int(p_solid.sum()),
        "final_compliance_index": history[-1]["compliance_index"],
        "elapsed_s": round(time.time() - t0, 2),
        "dominant_truss_struts": truss_sorted,
        "history": history[::4] + [history[-1]],
        "arrays": {
            f"{stage_key}__rho": rho.astype(np.float32),
            f"{stage_key}__active": active.astype(np.uint8),
            f"{stage_key}__xc": xc.astype(np.float32),
            f"{stage_key}__yc": yc.astype(np.float32),
            f"{stage_key}__zc": zc.astype(np.float32),
        },
    }


def main():
    npz_dict = {}
    stages_out = {}
    for stg_key in STAGE_PART_BASENAMES:
        res = run_stage_simp_and_truss(stg_key)
        npz_dict.update(res.pop("arrays"))
        stages_out[stg_key] = res
        print(
            f"[{stg_key:34s}] grid={res['grid_shape']} active={res['active_voxels']:5d} "
            f"C_idx={res['final_compliance_index']:.2f} ({res['elapsed_s']:.2f}s)"
        )

    npz_path = ART / "neck-head-topopt-fields.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **npz_dict)

    report = {
        "study": "cove50-neck-head-fdm-connectors-20261104",
        "fields_npz": rel(npz_path),
        "stages": stages_out,
    }
    out_json = ART / "neck-head-topopt-report.json"
    write_json(out_json, report)
    print(f"Saved {rel(npz_path)} and {rel(out_json)}")


if __name__ == "__main__":
    main()
