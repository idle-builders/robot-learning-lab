"""Run 3D blank-space SIMP topology optimization for the 3 lower-leg connector stages.

Starts from pure 3D design domains bounded only by the fixed motor/horn/sole interfaces,
swept kinematic keep-out zones, and shell interior (zero bent-sheet seed), and solves
multi-load 3D finite-element SIMP optimization with spatial density filtering.
"""
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from common import ART, write_json, ref


def hex8_ke(E: float = 1.0, nu: float = 0.33) -> np.ndarray:
    """Analytical 24x24 stiffness matrix for a unit cube 8-node trilinear hexahedron."""
    pts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=float)
    gp = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    g = (1.0 - 2.0 * nu) / 2.0
    D = c * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, g, 0, 0],
        [0, 0, 0, 0, g, 0],
        [0, 0, 0, 0, 0, g],
    ])
    KE = np.zeros((24, 24), dtype=float)
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN = 0.125 * np.array([
                    [s[0] * (1 + s[1] * eta) * (1 + s[2] * zeta) for s in pts],
                    [s[1] * (1 + s[0] * xi) * (1 + s[2] * zeta) for s in pts],
                    [s[2] * (1 + s[0] * xi) * (1 + s[1] * eta) for s in pts],
                ])  #J = 0.5*I for unit cube, so dN_dx = 2*dN, detJ = 0.125
                dN_dx = 2.0 * dN
                B = np.zeros((6, 24), dtype=float)
                for i in range(8):
                    bx, by, bz = dN_dx[:, i]
                    B[0, 3 * i] = bx
                    B[1, 3 * i + 1] = by
                    B[2, 3 * i + 2] = bz
                    B[3, 3 * i] = by
                    B[3, 3 * i + 1] = bx
                    B[4, 3 * i + 1] = bz
                    B[4, 3 * i + 2] = by
                    B[5, 3 * i] = bz
                    B[5, 3 * i + 2] = bx
                KE += (B.T @ D @ B) * 0.125
    return KE


def solve_simp_3d(
    active_mask: np.ndarray,
    passive_solid: np.ndarray,
    fixed_nodal_dofs: np.ndarray,
    load_cases: list[np.ndarray],
    volfrac: float = 0.32,
    penal: float = 3.0,
    rmin_vox: float = 1.35,
    n_iter: int = 24,
):
    """Run 3D multi-load SIMP topology optimization on an active voxel domain."""
    nx, ny, nz = active_mask.shape
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    n_nodes = nnx * nny * nnz
    n_dofs = 3 * n_nodes

    def nid(ix, iy, iz):
        return ix * (nny * nnz) + iy * nnz + iz

    ex, ey, ez = np.nonzero(active_mask)
    n_el = len(ex)
    n000 = nid(ex, ey, ez)
    n100 = nid(ex + 1, ey, ez)
    n110 = nid(ex + 1, ey + 1, ez)
    n010 = nid(ex, ey + 1, ez)
    n001 = nid(ex, ey, ez + 1)
    n101 = nid(ex + 1, ey, ez + 1)
    n111 = nid(ex + 1, ey + 1, ez + 1)
    n011 = nid(ex, ey + 1, ez + 1)
    enodes = np.column_stack([n000, n100, n110, n010, n001, n101, n111, n011])
    edofs = np.empty((n_el, 24), dtype=np.int32)
    for k in range(8):
        edofs[:, 3 * k] = 3 * enodes[:, k]
        edofs[:, 3 * k + 1] = 3 * enodes[:, k] + 1
        edofs[:, 3 * k + 2] = 3 * enodes[:, k] + 2

    KE = hex8_ke(1.0, 0.33)
    iK = np.repeat(edofs, 24, axis=1).ravel()
    jK = np.tile(edofs, (1, 24)).ravel()

    # Restrict DOFs to active nodes that aren't fixed
    active_nodes = np.unique(enodes.ravel())
    active_dofs = np.sort(np.r_[3 * active_nodes, 3 * active_nodes + 1, 3 * active_nodes + 2])
    free_dofs = np.setdiff1d(active_dofs, fixed_nodal_dofs)

    passive_el = passive_solid[ex, ey, ez]
    x = np.full(n_el, volfrac, dtype=float)
    x[passive_el] = 1.0
    xPhys = x.copy()

    E0 = 1.0
    Emin = 1e-4
    history = []

    for it in range(1, n_iter + 1):
        t0 = time.time()
        penal_cur = min(penal, 1.5 + (penal - 1.5) * (it - 1) / max(1, n_iter - 6))
        Ee = Emin + (xPhys ** penal_cur) * (E0 - Emin)
        sK = (Ee[:, None, None] * KE[None, :, :]).ravel()
        K = coo_matrix((sK, (iK, jK)), shape=(n_dofs, n_dofs)).tocsr()
        K_ff = K[free_dofs][:, free_dofs]
        diag_K = K_ff.diagonal()
        M_prec = diags(1.0 / np.maximum(diag_K, 1e-9))

        ce_total = np.zeros(n_el, dtype=float)
        total_compliance = 0.0
        for F in load_cases:
            U = np.zeros(n_dofs, dtype=float)
            f_free = F[free_dofs]
            u_free, _ = cg(K_ff, f_free, M=M_prec, rtol=2e-4, maxiter=350)
            U[free_dofs] = u_free
            total_compliance += float(F @ U)
            Ue = U[edofs]
            ce_total += np.einsum("ei,ij,ej->e", Ue, KE, Ue)

        dc = -penal_cur * (xPhys ** (penal_cur - 1.0)) * (E0 - Emin) * ce_total

        # Spatial Gaussian/Helmholtz filter on 3D grid
        dc_grid = np.zeros_like(active_mask, dtype=float)
        w_grid = np.zeros_like(active_mask, dtype=float)
        dc_grid[ex, ey, ez] = dc * xPhys
        w_grid[ex, ey, ez] = xPhys
        dc_filt = gaussian_filter(dc_grid, sigma=rmin_vox, mode="nearest")
        w_filt = np.maximum(gaussian_filter(w_grid, sigma=rmin_vox, mode="nearest"), 1e-9)
        dc = dc_filt[ex, ey, ez] / w_filt[ex, ey, ez]

        # Optimality criteria (OC) update
        l1, l2 = 0.0, 1e9
        move = 0.18
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(np.maximum(-dc, 0.0) / lmid)))))
            xnew[passive_el] = 1.0
            # Filter density
            x_grid = np.zeros_like(active_mask, dtype=float)
            m_grid = active_mask.astype(float)
            x_grid[ex, ey, ez] = xnew
            xf = gaussian_filter(x_grid, sigma=0.75 * rmin_vox, mode="nearest") / np.maximum(gaussian_filter(m_grid, sigma=0.75 * rmin_vox, mode="nearest"), 1e-9)
            xPhys_cand = xf[ex, ey, ez]
            xPhys_cand[passive_el] = 1.0
            if xPhys_cand.mean() > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = xnew
        xPhys = xPhys_cand
        history.append({
            "iter": it,
            "compliance": round(total_compliance, 4),
            "volfrac": round(float(xPhys.mean()), 4),
            "max_change": round(change, 4),
            "sec": round(time.time() - t0, 2),
        })

    rho_grid = np.zeros_like(active_mask, dtype=np.float32)
    rho_grid[ex, ey, ez] = xPhys.astype(np.float32)
    return rho_grid, history


def build_and_solve_all():
    out_dir = ART / "topopt"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    fields = {}

    # =========================================================================
    # STAGE 1: SHIN BLANK-SPACE TOPOLOGY OPTIMIZATION (R_shin)
    # =========================================================================
    # Voxel resolution h = 1.5 mm over X[39.25..81.25], Y[-11.2..17.6], Z[65.5..133.0]
    hx = 1.5
    xs = np.arange(39.25, 81.25, hx)
    ys = np.arange(-11.2, 17.6, hx)
    zs = np.arange(65.5, 133.0, hx)
    X, Y, Z = np.meshgrid(xs + 0.5 * hx, ys + 0.5 * hx, zs + 0.5 * hx, indexing="ij")

    # Blank design domain: cheeks + upper bridge corridor + anterior arch corridor + motor standoffs
    in_inner_cheek = (X >= 39.25) & (X <= 42.5) & (Y >= -11.2) & (Y <= 17.5) & (Z >= 66.0) & (Z <= 133.0)
    in_outer_cheek = (X >= 77.5) & (X <= 81.25) & (Y >= -11.2) & (Y <= 17.5) & (Z >= 66.0) & (Z <= 133.0)
    in_upper_bridge = (X >= 41.25) & (X <= 78.75) & (Y >= -11.2) & (Y <= 8.5) & (Z >= 96.5) & (Z <= 106.5)
    in_front_arch = (X >= 41.25) & (X <= 78.75) & (Y >= -11.2) & (Y <= -8.5) & (Z >= 72.0) & (Z <= 106.5)
    in_standoffs = (
        (((X >= 41.25) & (X <= 44.2)) | ((X >= 75.8) & (X <= 78.75)))
        & (
            ((np.hypot(Y - (-6.25), Z - 69.3) <= 4.5))
            | ((np.hypot(Y - 14.25, Z - 92.0) <= 5.5))
        )
    )
    active_shin = in_inner_cheek | in_outer_cheek | in_upper_bridge | in_front_arch | in_standoffs
    # Keep-outs: knee horn central bore (r=4.2 at Y=4, Z=123), ankle horn window (r=11.0 at Y=4, Z=61),
    # upper thigh flexion corner (Y > 12.0 & Z > 108.0 or Y < -4.0 & Z > 108.0)
    keepout_shin = (
        (np.hypot(Y - 4.0, Z - 123.0) < 4.2)
        | (np.hypot(Y - 4.0, Z - 61.0) < 11.2)
        | ((Z > 108.0) & ((Y > 12.0) | (Y < -4.0)))
    )
    active_shin &= ~keepout_shin

    passive_shin = (
        ((np.hypot(Y - 4.0, Z - 116.0) <= 3.5) | (np.hypot(Y - 4.0, Z - 130.0) <= 3.5))
        & ((X <= 42.5) | (X >= 77.5))
    ) | in_standoffs

    # Nodes & BCs for Stage 1
    nnx, nny, nnz = len(xs) + 1, len(ys) + 1, len(zs) + 1
    ix_g, iy_g, iz_g = np.meshgrid(np.arange(nnx), np.arange(nny), np.arange(nnz), indexing="ij")
    XN = 39.25 + ix_g * hx
    YN = -11.2 + iy_g * hx
    ZN = 65.5 + iz_g * hx
    nid_grid = ix_g * (nny * nnz) + iy_g * nnz + iz_g

    fixed_nodes_shin = nid_grid[
        ((np.hypot(YN - 4.0, ZN - 116.0) <= 3.2) | (np.hypot(YN - 4.0, ZN - 130.0) <= 3.2))
        & ((XN <= 42.5) | (XN >= 77.5))
    ].ravel()
    fixed_dofs_shin = np.sort(np.r_[3 * fixed_nodes_shin, 3 * fixed_nodes_shin + 1, 3 * fixed_nodes_shin + 2])

    load_nodes_shin = nid_grid[
        (((XN >= 41.25) & (XN <= 44.5)) | ((XN >= 75.5) & (XN <= 78.75)))
        & ((np.hypot(YN - (-6.25), ZN - 69.3) <= 3.8) | (np.hypot(YN - 14.25, ZN - 92.0) <= 4.5))
    ].ravel()
    n_dofs_shin = 3 * nnx * nny * nnz
    loads_shin = []
    # Case 1: Vertical stance + roll bending moment
    f1 = np.zeros(n_dofs_shin)
    f1[3 * load_nodes_shin + 2] = 1.0 / len(load_nodes_shin)
    f1[3 * load_nodes_shin + 0] = 0.6 * ((ZN.ravel()[load_nodes_shin] - 80.0) / 15.0) / len(load_nodes_shin)
    loads_shin.append(f1)
    # Case 2: AP shear + pitch bending
    f2 = np.zeros(n_dofs_shin)
    f2[3 * load_nodes_shin + 1] = 1.0 / len(load_nodes_shin)
    f2[3 * load_nodes_shin + 2] = 0.6 * ((YN.ravel()[load_nodes_shin] - 4.0) / 12.0) / len(load_nodes_shin)
    loads_shin.append(f2)
    # Case 3: Yaw torsion + ML shear
    f3 = np.zeros(n_dofs_shin)
    f3[3 * load_nodes_shin + 0] = 1.0 / len(load_nodes_shin)
    f3[3 * load_nodes_shin + 1] = 0.8 * ((XN.ravel()[load_nodes_shin] - 60.0) / 18.0) / len(load_nodes_shin)
    loads_shin.append(f3)

    rho_shin, hist_shin = solve_simp_3d(active_shin, passive_shin, fixed_dofs_shin, loads_shin, volfrac=0.34, rmin_vox=1.3, n_iter=22)
    fields["shin_rho"] = rho_shin
    fields["shin_axes_mm"] = np.array([xs[0], xs[-1] + hx, ys[0], ys[-1] + hx, zs[0], zs[-1] + hx])
    reports["stage1_shin"] = {
        "grid_shape": list(active_shin.shape),
        "voxel_mm": hx,
        "active_voxels": int(active_shin.sum()),
        "target_volfrac": 0.34,
        "history": hist_shin,
    }
    print("Stage 1 Shin TopOpt done:", hist_shin[-1], flush=True)

    # =========================================================================
    # STAGE 2: ANKLE COUPLER BLANK-SPACE TOPOLOGY OPTIMIZATION (R_ankle_pitch)
    # =========================================================================
    hx2 = 1.25
    xs2 = np.arange(39.0, 83.5, hx2)
    ys2 = np.arange(-6.5, 22.5, hx2)
    zs2 = np.arange(8.5, 71.5, hx2)
    X2, Y2, Z2 = np.meshgrid(xs2 + 0.5 * hx2, ys2 + 0.5 * hx2, zs2 + 0.5 * hx2, indexing="ij")

    in_main_cheek = (X2 >= 78.5) & (X2 <= 83.0) & (np.hypot(Y2 - 4.0, Z2 - 61.0) <= 11.2)
    in_idler_cheek = (X2 >= 39.0) & (X2 <= 42.5) & (np.hypot(Y2 - 4.0, Z2 - 61.0) <= 10.8) & (Y2 >= -2.0)
    in_roll_pad = (Y2 >= 17.5) & (Y2 <= 22.0) & (np.hypot(X2 - 75.0, Z2 - 19.0) <= 11.2)
    in_arch_bridge = (X2 >= 39.5) & (X2 <= 82.8) & (Y2 >= 14.5) & (Y2 <= 22.2) & (Z2 >= 26.0) & (Z2 <= 58.0)
    in_outer_strut = (X2 >= 76.0) & (X2 <= 82.8) & (Y2 >= -3.0) & (Y2 <= 22.0) & (Z2 >= 24.0) & (Z2 <= 62.0)
    active_coupler = in_main_cheek | in_idler_cheek | in_roll_pad | in_arch_bridge | in_outer_strut

    # Keep-outs: swept ankle pitch motor case & swept foot roll motor case & central horn bores
    keepout_coupler = (
        ((X2 > 42.2) & (X2 < 78.8) & (Y2 < 15.2) & (Z2 > 45.0))
        | ((X2 > 40.0) & (X2 < 85.0) & (Y2 < 17.2) & (Z2 < 34.5))
        | ((X2 > 41.0) & (X2 < 64.0) & (Y2 < 16.5) & (Z2 < 42.0))
        | (np.hypot(Y2 - 4.0, Z2 - 61.0) < 4.5)
        | (np.hypot(X2 - 75.0, Z2 - 19.0) < 4.5)
    )
    active_coupler &= ~keepout_coupler
    passive_coupler = (in_main_cheek | in_idler_cheek | in_roll_pad) & active_coupler

    nnx2, nny2, nnz2 = len(xs2) + 1, len(ys2) + 1, len(zs2) + 1
    ix2, iy2, iz2 = np.meshgrid(np.arange(nnx2), np.arange(nny2), np.arange(nnz2), indexing="ij")
    XN2 = xs2[0] + ix2 * hx2
    YN2 = ys2[0] + iy2 * hx2
    ZN2 = zs2[0] + iz2 * hx2
    nid2 = ix2 * (nny2 * nnz2) + iy2 * nnz2 + iz2

    fixed_nodes_c = nid2[
        (((XN2 >= 78.5) & (np.hypot(YN2 - 4.0, ZN2 - 61.0) <= 9.0))
         | ((XN2 <= 42.5) & (np.hypot(YN2 - 4.0, ZN2 - 61.0) <= 8.5) & (YN2 >= 0.0)))
        & (np.hypot(YN2 - 4.0, ZN2 - 61.0) >= 4.5)
    ].ravel()
    fixed_dofs_c = np.sort(np.r_[3 * fixed_nodes_c, 3 * fixed_nodes_c + 1, 3 * fixed_nodes_c + 2])

    load_nodes_c = nid2[
        (YN2 >= 17.5) & (np.hypot(XN2 - 75.0, ZN2 - 19.0) <= 9.0) & (np.hypot(XN2 - 75.0, ZN2 - 19.0) >= 4.5)
    ].ravel()
    n_dofs_c = 3 * nnx2 * nny2 * nnz2
    loads_c = []
    fc1 = np.zeros(n_dofs_c)
    fc1[3 * load_nodes_c + 2] = 1.0 / len(load_nodes_c)
    fc1[3 * load_nodes_c + 1] = -0.5 * ((ZN2.ravel()[load_nodes_c] - 19.0) / 7.0) / len(load_nodes_c)
    loads_c.append(fc1)
    fc2 = np.zeros(n_dofs_c)
    fc2[3 * load_nodes_c + 0] = 0.8 / len(load_nodes_c)
    fc2[3 * load_nodes_c + 1] = 0.8 / len(load_nodes_c)
    loads_c.append(fc2)
    fc3 = np.zeros(n_dofs_c)
    fc3[3 * load_nodes_c + 2] = ((XN2.ravel()[load_nodes_c] - 75.0) / 7.0) / len(load_nodes_c)
    loads_c.append(fc3)

    rho_coupler, hist_coupler = solve_simp_3d(active_coupler, passive_coupler, fixed_dofs_c, loads_c, volfrac=0.35, rmin_vox=1.3, n_iter=22)
    fields["coupler_rho"] = rho_coupler
    fields["coupler_axes_mm"] = np.array([xs2[0], xs2[-1] + hx2, ys2[0], ys2[-1] + hx2, zs2[0], zs2[-1] + hx2])
    reports["stage2_ankle_coupler"] = {
        "grid_shape": list(active_coupler.shape),
        "voxel_mm": hx2,
        "active_voxels": int(active_coupler.sum()),
        "target_volfrac": 0.35,
        "history": hist_coupler,
    }
    print("Stage 2 Ankle Coupler TopOpt done:", hist_coupler[-1], flush=True)

    # =========================================================================
    # STAGE 3: FOOT MONOCOQUE SOLE & CRADLE TOPOLOGY OPTIMIZATION (R_foot)
    # =========================================================================
    hx3 = 1.8
    xs3 = np.arange(37.0, 97.0, hx3)
    ys3 = np.arange(-55.0, 31.0, hx3)
    zs3 = np.arange(2.0, 15.5, hx3)
    X3, Y3, Z3 = np.meshgrid(xs3 + 0.5 * hx3, ys3 + 0.5 * hx3, zs3 + 0.5 * hx3, indexing="ij")

    # Rounded rectangle 60 x 86, R=22 centered at (67, -12)
    dx_r = np.maximum(np.abs(X3 - 67.0) - (30.0 - 22.0), 0.0)
    dy_r = np.maximum(np.abs(Y3 - (-12.0)) - (43.0 - 22.0), 0.0)
    in_sole_outline = np.hypot(dx_r, dy_r) <= 22.0

    in_sole_plate = in_sole_outline & (Z3 <= 6.2)
    in_front_saddle = in_sole_outline & (X3 >= 39.0) & (X3 <= 70.0) & (Y3 >= 14.8) & (Y3 <= 28.0) & (Z3 <= 15.0)
    in_rear_saddle = in_sole_outline & (X3 >= 39.0) & (X3 <= 70.0) & (Y3 >= -30.5) & (Y3 <= -17.2) & (Z3 <= 15.0)
    in_medial_arch = in_sole_outline & (X3 >= 37.2) & (X3 <= 40.0) & (Y3 >= -26.0) & (Y3 <= 24.0) & (Z3 <= 14.5)
    active_foot = in_sole_plate | in_front_saddle | in_rear_saddle | in_medial_arch

    # Keep-out for motor case & roll horn clearance
    keepout_foot = (
        ((X3 > 39.8) & (X3 < 85.5) & (Y3 > -17.2) & (Y3 < 14.8) & (Z3 > 6.2))
        | (np.hypot(X3 - 75.0, Z3 - 19.0) < 11.2)
    )
    active_foot &= ~keepout_foot

    # Passive solid: bottom skin Z <= 3.5 under sensors + motor mounting bosses at Z=8.75
    in_sensor_zones = (
        ((X3 >= 70.5) & (X3 <= 89.5) & (Y3 >= -47.5) & (Y3 <= -28.5))
        | ((X3 >= 70.5) & (X3 <= 89.5) & (Y3 >= 2.5) & (Y3 <= 21.5))
        | ((X3 >= 39.5) & (X3 <= 58.5) & (Y3 >= -14.5) & (Y3 <= 4.5))
    ) & (Z3 <= 4.0)
    in_motor_bosses = (
        ((Y3 >= 14.8) & (Y3 <= 20.0) & ((np.hypot(X3 - 46.0, Z3 - 8.75) <= 3.2) | (np.hypot(X3 - 66.7, Z3 - 8.75) <= 3.2)))
        | ((Y3 >= -22.5) & (Y3 <= -17.2) & ((np.hypot(X3 - 42.25, Z3 - 8.75) <= 3.2) | (np.hypot(X3 - 66.7, Z3 - 8.75) <= 3.2)))
    )
    passive_foot = (in_sensor_zones | in_motor_bosses) & active_foot

    nnx3, nny3, nnz3 = len(xs3) + 1, len(ys3) + 1, len(zs3) + 1
    ix3, iy3, iz3 = np.meshgrid(np.arange(nnx3), np.arange(nny3), np.arange(nnz3), indexing="ij")
    XN3 = xs3[0] + ix3 * hx3
    YN3 = ys3[0] + iy3 * hx3
    ZN3 = zs3[0] + iz3 * hx3
    nid3 = ix3 * (nny3 * nnz3) + iy3 * nnz3 + iz3

    fixed_nodes_f = nid3[
        ((YN3 >= 14.5) & (YN3 <= 20.5) & ((np.hypot(XN3 - 46.0, ZN3 - 8.75) <= 3.0) | (np.hypot(XN3 - 66.7, ZN3 - 8.75) <= 3.0)))
        | ((YN3 >= -23.0) & (YN3 <= -17.0) & ((np.hypot(XN3 - 42.25, ZN3 - 8.75) <= 3.0) | (np.hypot(XN3 - 66.7, ZN3 - 8.75) <= 3.0)))
    ].ravel()
    fixed_dofs_f = np.sort(np.r_[3 * fixed_nodes_f, 3 * fixed_nodes_f + 1, 3 * fixed_nodes_f + 2])

    # Ground reaction load cases across toe, heel, medial, and lateral sole patches at Z=2.0
    n_dofs_f = 3 * nnx3 * nny3 * nnz3
    loads_f = []
    for x_c, y_c in [(78.0, -38.0), (78.0, 12.0), (49.0, -5.0), (50.0, -35.0), (50.0, 15.0)]:
        patch = nid3[(ZN3 <= 2.5) & (np.hypot(XN3 - x_c, YN3 - y_c) <= 10.0)].ravel()
        ff = np.zeros(n_dofs_f)
        ff[3 * patch + 2] = 1.0 / max(1, len(patch))
        ff[3 * patch + 0] = 0.25 / max(1, len(patch))
        ff[3 * patch + 1] = 0.25 / max(1, len(patch))
        loads_f.append(ff)

    rho_foot, hist_foot = solve_simp_3d(active_foot, passive_foot, fixed_dofs_f, loads_f, volfrac=0.32, rmin_vox=1.3, n_iter=22)
    fields["foot_rho"] = rho_foot
    fields["foot_axes_mm"] = np.array([xs3[0], xs3[-1] + hx3, ys3[0], ys3[-1] + hx3, zs3[0], zs3[-1] + hx3])
    reports["stage3_foot_monocoque"] = {
        "grid_shape": list(active_foot.shape),
        "voxel_mm": hx3,
        "active_voxels": int(active_foot.sum()),
        "target_volfrac": 0.32,
        "history": hist_foot,
    }
    print("Stage 3 Foot Monocoque TopOpt done:", hist_foot[-1], flush=True)

    np.savez_compressed(out_dir / "topopt-fields.npz", **fields)
    write_json(out_dir / "topopt-solve-report.json", {
        "study": "cove50-lower-leg-fdm-topopt",
        "method": "3D Hex8 multi-load SIMP topology optimization with spatial density filtering from blank design domains",
        "stages": reports,
        "fields_file": ref(out_dir / "topopt-fields.npz"),
        "source_code": ref(Path(__file__)),
    })


if __name__ == "__main__":
    build_and_solve_all()
