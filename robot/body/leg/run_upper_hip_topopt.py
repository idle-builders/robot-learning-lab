"""Run 3D blank-space SIMP topology optimization for the 3 upper-leg & hip connector stages:
- Stage 4 (`R/L_roll_link` — Thigh: Knee C047 Case -> Hip Roll Horn + Rear Idler)
- Stage 5 (`R/L_pitch_link` — Hip Pitch-to-Roll Yoke: Hip Roll C047 Case -> Hip Pitch Horn + Idler)
- Stage 6 (`R/L_yaw_link` — Hip Yaw-to-Pitch Frame: Hip Pitch C047 Case -> Hip Yaw Horn + Idler)
"""
from pathlib import Path
import time
import numpy as np
from common import ART, write_json
from run_topopt_field import solve_simp_3d


def make_grid(x_range, y_range, z_range, dx: float):
    xs = np.arange(x_range[0], x_range[1] + dx, dx)
    ys = np.arange(y_range[0], y_range[1] + dx, dx)
    zs = np.arange(z_range[0], z_range[1] + dx, dx)
    xc = xs[:-1] + 0.5 * dx
    yc = ys[:-1] + 0.5 * dx
    zc = zs[:-1] + 0.5 * dx
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    return xs, ys, zs, X, Y, Z


def apply_wrench_to_nodes(
    xs, ys, zs, mask_el: np.ndarray, force_N: tuple[float, float, float], moment_Nmm: tuple[float, float, float], ref_pt_mm: tuple[float, float, float]
) -> np.ndarray:
    nx, ny, nz = mask_el.shape
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    n_dofs = 3 * nnx * nny * nnz
    F = np.zeros(n_dofs, dtype=float)

    ex, ey, ez = np.nonzero(mask_el)
    nodes = set()
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                for ix, iy, iz in zip(ex + dx, ey + dy, ez + dz):
                    nodes.add((int(ix), int(iy), int(iz)))
    nodes = sorted(nodes)
    coords = np.array([[xs[i], ys[j], zs[k]] for i, j, k in nodes], dtype=float)
    r = coords - np.asarray(ref_pt_mm, dtype=float)
    n_n = len(nodes)
    f_direct = np.asarray(force_N, dtype=float) / n_n
    r2 = np.sum(r ** 2, axis=1, keepdims=True)
    M = np.asarray(moment_Nmm, dtype=float)
    f_mom = np.cross(M[None, :], r) / max(float(np.sum(r2)), 1.0)
    f_tot = f_direct[None, :] + f_mom

    for idx, (i, j, k) in enumerate(nodes):
        nid = i * (nny * nnz) + j * nnz + k
        F[3 * nid : 3 * nid + 3] += f_tot[idx]
    return F


def fixed_dofs_from_elements(mask_el: np.ndarray) -> np.ndarray:
    nx, ny, nz = mask_el.shape
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    ex, ey, ez = np.nonzero(mask_el)
    dofs = set()
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                for ix, iy, iz in zip(ex + dx, ey + dy, ez + dz):
                    nid = int(ix) * (nny * nnz) + int(iy) * nnz + int(iz)
                    dofs.update((3 * nid, 3 * nid + 1, 3 * nid + 2))
    return np.array(sorted(dofs), dtype=np.int32)


def solve_stage4_thigh():
    """Stage 4: Thigh connector between Knee C047 motor case and Hip Roll horn + rear idler."""
    dx = 2.0
    xs, ys, zs, X, Y, Z = make_grid((40.0, 78.0), (-16.0, 30.0), (128.0, 202.0), dx)

    # Keep out of Knee C047 motor body and Hip Roll C047 motor body
    knee_motor = (X >= 44.5) & (Y >= -8.8) & (Y <= 16.8) & (Z <= 158.5)
    roll_motor = (X >= 28.5) & (X <= 74.5) & (Y >= -11.8) & (Y <= 25.0) & (Z >= 175.0)

    corridor = (
        ((X <= 46.8) & (Y >= -14.2) & (Y <= 27.5) & (Z >= 129.0) & (Z <= 174.0))
        | ((Y <= -11.5) & (X >= 44.0) & (X <= 77.0) & (Z >= 146.0) & (Z <= 200.0))
        | ((Y >= 24.5) & (X >= 44.0) & (X <= 77.0) & (Z >= 154.0) & (Z <= 200.0))
        | ((Z >= 158.5) & (Z <= 174.0) & (X >= 42.5) & (X <= 76.5) & (Y >= -14.2) & (Y <= 27.5))
    )
    active = corridor & (~knee_motor) & (~roll_motor)

    # Passive bosses: 4 Knee PA2 bolts on medial cheek + 4 front Hip Roll horn bolts + 4 rear Hip Roll idler bolts
    knee_bosses = np.zeros_like(active)
    for by, bz in [(-6.25, 131.3), (-6.25, 152.0), (14.25, 131.3), (14.25, 152.0)]:
        knee_bosses |= (X <= 46.5) & ((Y - by) ** 2 + (Z - bz) ** 2 <= 4.8 ** 2)

    roll_front_bosses = np.zeros_like(active)
    roll_rear_bosses = np.zeros_like(active)
    for bx, bz in [(57.0, 190.0), (71.0, 190.0), (64.0, 183.0), (64.0, 197.0)]:
        roll_front_bosses |= (Y <= -11.5) & ((X - bx) ** 2 + (Z - bz) ** 2 <= 4.8 ** 2)
        roll_rear_bosses |= (Y >= 24.5) & ((X - bx) ** 2 + (Z - bz) ** 2 <= 4.8 ** 2)

    passive = active & (knee_bosses | roll_front_bosses | roll_rear_bosses)
    fixed_dofs = fixed_dofs_from_elements(active & (roll_front_bosses | roll_rear_bosses))
    load_el = active & knee_bosses

    lc1 = apply_wrench_to_nodes(xs, ys, zs, load_el, (22.0, -28.0, 65.0), (3100.0, 1650.0, 520.0), (60.0, 4.0, 140.0))
    lc2 = apply_wrench_to_nodes(xs, ys, zs, load_el, (-18.0, 24.0, 58.0), (-2400.0, 1800.0, -480.0), (60.0, 4.0, 140.0))

    rho, hist = solve_simp_3d(active, passive, fixed_dofs, [lc1, lc2], volfrac=0.34, penal=3.0, rmin_vox=1.25, n_iter=20)
    return {"xs": xs, "ys": ys, "zs": zs, "rho": rho, "active": active, "history": hist}


def solve_stage5_pitch_to_roll():
    """Stage 5: Hip Pitch-to-Roll yoke between Hip Roll C047 case and Hip Pitch horn + idler."""
    dx = 2.0
    xs, ys, zs, X, Y, Z = make_grid((24.0, 71.0), (-13.0, 27.0), (176.0, 231.0), dx)

    roll_motor = (X >= 28.5) & (X <= 74.5) & (Y >= -8.8) & (Y <= 23.2) & (Z <= 202.8)
    pitch_motor = (X <= 58.8) & (Y >= -10.5) & (Y <= 14.5) & (Z >= 208.0)

    corridor = (
        ((Y <= -8.5) & (X >= 26.0) & (X <= 69.5) & (Z >= 177.0) & (Z <= 224.0))
        | ((Y >= 22.8) & (X >= 26.0) & (X <= 69.5) & (Z >= 177.0) & (Z <= 230.0))
        | ((X >= 59.0) & (X <= 69.5) & (Y >= -11.0) & (Y <= 25.5) & (Z >= 198.0) & (Z <= 230.0))
    )
    active = corridor & (~roll_motor) & (~pitch_motor)

    front_roll_bosses = np.zeros_like(active)
    for bx, bz in [(35.0, 179.75), (35.0, 200.25), (55.7, 179.75), (55.7, 200.25)]:
        front_roll_bosses |= (Y <= -8.5) & ((X - bx) ** 2 + (Z - bz) ** 2 <= 4.5 ** 2)

    rear_roll_bosses = np.zeros_like(active)
    for bx, bz in [(31.25, 179.75), (31.25, 200.25), (55.7, 179.75), (55.7, 200.25)]:
        rear_roll_bosses |= (Y >= 22.8) & ((X - bx) ** 2 + (Z - bz) ** 2 <= 4.5 ** 2)

    pitch_horn_bosses = np.zeros_like(active)
    for by, bz in [(2.0, 212.0), (2.0, 226.0), (-5.0, 219.0), (9.0, 219.0)]:
        pitch_horn_bosses |= (X >= 59.0) & ((Y - by) ** 2 + (Z - bz) ** 2 <= 4.8 ** 2)

    passive = active & (front_roll_bosses | rear_roll_bosses | pitch_horn_bosses)
    fixed_dofs = fixed_dofs_from_elements(active & pitch_horn_bosses)
    load_el = active & (front_roll_bosses | rear_roll_bosses)

    lc1 = apply_wrench_to_nodes(xs, ys, zs, load_el, (24.0, -30.0, 68.0), (3400.0, 1950.0, 620.0), (50.0, 7.0, 190.0))
    lc2 = apply_wrench_to_nodes(xs, ys, zs, load_el, (-20.0, 26.0, 60.0), (-2800.0, 2100.0, -560.0), (50.0, 7.0, 190.0))

    rho, hist = solve_simp_3d(active, passive, fixed_dofs, [lc1, lc2], volfrac=0.35, penal=3.0, rmin_vox=1.25, n_iter=20)
    return {"xs": xs, "ys": ys, "zs": zs, "rho": rho, "active": active, "history": hist}


def solve_stage6_yaw_to_pitch():
    """Stage 6: Hip Yaw-to-Pitch frame between Hip Pitch C047 case and Hip Yaw horn."""
    dx = 1.8
    xs, ys, zs, X, Y, Z = make_grid((48.0, 78.0), (-16.0, 20.0), (224.0, 275.5), dx)

    pitch_motor = (X <= 55.8) & (Y >= -10.5) & (Y <= 14.5) & (Z <= 254.5)
    center_cavity = (X >= 55.5) & (X <= 62.5) & (Y >= -5.2) & (Y <= 9.2) & (Z <= 258.5)

    corridor = (
        ((Z >= 261.0) & (Z <= 274.75) & (X >= 49.0) & (X <= 76.0) & (Y >= -15.0) & (Y <= 19.0))
        | ((X >= 55.8) & (X <= 74.0) & (Y >= -15.0) & (Y <= 19.0) & (Z >= 225.4) & (Z <= 266.75))
    )
    active = corridor & (~pitch_motor) & (~center_cavity)

    yaw_horn_bosses = np.zeros_like(active)
    for bx, by in [(52.0, 2.0), (66.0, 2.0), (59.0, -5.0), (59.0, 9.0)]:
        yaw_horn_bosses |= (Z >= 261.0) & ((X - bx) ** 2 + (Y - by) ** 2 <= 4.5 ** 2)

    pitch_case_bosses = np.zeros_like(active)
    for by, bz in [(-8.25, 227.3), (-8.25, 248.0), (12.25, 227.3), (12.25, 248.0)]:
        pitch_case_bosses |= (X <= 62.0) & ((Y - by) ** 2 + (Z - bz) ** 2 <= 4.5 ** 2)

    passive = active & (yaw_horn_bosses | pitch_case_bosses)
    fixed_dofs = fixed_dofs_from_elements(active & yaw_horn_bosses)
    load_el = active & pitch_case_bosses

    lc1 = apply_wrench_to_nodes(xs, ys, zs, load_el, (26.0, -32.0, 72.0), (3800.0, 2200.0, 740.0), (57.5, 2.0, 238.0))
    lc2 = apply_wrench_to_nodes(xs, ys, zs, load_el, (-22.0, 28.0, 64.0), (-3200.0, 2300.0, -680.0), (57.5, 2.0, 238.0))

    rho, hist = solve_simp_3d(active, passive, fixed_dofs, [lc1, lc2], volfrac=0.35, penal=3.0, rmin_vox=1.25, n_iter=20)
    return {"xs": xs, "ys": ys, "zs": zs, "rho": rho, "active": active, "history": hist}


def main():
    t0 = time.time()
    s4 = solve_stage4_thigh()
    s5 = solve_stage5_pitch_to_roll()
    s6 = solve_stage6_yaw_to_pitch()

    out_dir = ART / "topopt"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "topopt-upper-hip-fields.npz",
        s4_xs=s4["xs"], s4_ys=s4["ys"], s4_zs=s4["zs"], s4_rho=s4["rho"],
        s5_xs=s5["xs"], s5_ys=s5["ys"], s5_zs=s5["zs"], s5_rho=s5["rho"],
        s6_xs=s6["xs"], s6_ys=s6["ys"], s6_zs=s6["zs"], s6_rho=s6["rho"],
    )
    report = {
        "elapsed_s": round(time.time() - t0, 2),
        "stage4_thigh": {"active_voxels": int(s4["active"].sum()), "history": s4["history"]},
        "stage5_hip_pitch_to_roll": {"active_voxels": int(s5["active"].sum()), "history": s5["history"]},
        "stage6_hip_yaw_to_pitch": {"active_voxels": int(s6["active"].sum()), "history": s6["history"]},
    }
    write_json(out_dir / "topopt-upper-hip-report.json", report)
    print(f"Solved 3D SIMP topology optimization for Stages 4, 5, and 6 in {report['elapsed_s']:.2f}s")


if __name__ == "__main__":
    main()
