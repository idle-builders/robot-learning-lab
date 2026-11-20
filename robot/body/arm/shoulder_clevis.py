"""Shoulder roll clevis (owner M-j7z6f9 as scoped 2026-11-20: roll servo only): room for the stage 1 U-yoke.

Stage 1 (R_shoulder_pitch_to_roll_combined, pitch link) should become a U around the roll servo: bar on the inner (-x)
end of the roll case between the pitch main horn (x 82.6) and the case (x 92), arms on both case faces gripping the
roll main horn (+y) and a new roll idler (-y). Before any B-rep is built, this measures the space such a part can use.

  space   free space in the pitch-link frame: torso parts (centre shells, pitch case) swept over R_shoulder_pitch
          (-60..150) and roll-link parts (roll case, upper-arm cover, yaw case + its hardware, cover screws) swept over
          R_shoulder_roll (-15..95), 1 deg steps, 0.5 mm voxels; distances to the swept sets, connectivity between the
          horn side and the idler side, occupancy of the owner's bar box, the material of the torso shells and the cover
          that the U would need, and a labelled figure (.cache/work/shoulder-views/shoulder-clevis-space.png).
          Numbers -> .cache/work/shoulder-clevis/space.json.

  python3 infra/tools/memguard.py --limit-gb 5 --timeout-s 900 -- .venv/bin/python -B robot/body/arm/shoulder_clevis.py space
Every result is a measurement (mm, mm3, %); nothing accepts or rejects.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "robot/body"))
from checks import bodylib as F  # noqa: E402

MANIFEST = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"
WORK = ROOT / ".cache/work/shoulder-clevis"
VIEWS = ROOT / ".cache/work/shoulder-views"
V = 0.5                                             # voxel (mm)
LO, HI = np.array([78.0, -26.0, 328.0]), np.array([122.0, 34.0, 390.0])
ROLL_O = np.array([106.0, 5.05, 360.0])
PITCH_YZ = (9.0, 352.0)
SOCKET_R = 18.8                                      # torso shell cut about the pitch axis (assemble_complete_fdm_body sh_cut)
BAR_BOX = dict(x=(86.0, 90.0), y=(-14.0, -10.0), z=(345.0, 368.0))   # owner sketch: bar between pitch horn and roll case
HORN_SEED, IDLER_SEED = (86.0, 18.0, 352.0), (106.0, -16.0, 352.0)

GROUPS = {   # name: (parts, joint, seen from the pitch link: torso inverse-rotated, roll link forward-rotated)
    "shells": (["center_front_shell", "center_back_shell"], "R_shoulder_pitch", True),
    "pitch_case": (["upper__R_pitch_C047_case"], "R_shoulder_pitch", True),
    "roll_case": (["upper__R_roll_C047_case"], "R_shoulder_roll", False),
    "cover": (["R_upper_arm_hard_cover"], "R_shoulder_roll", False),
    "yaw_and_hardware": (None, "R_shoulder_roll", False),
}


def rotm(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def part_points(m, solid, crop):
    """Points of a part: a filled voxel cloud for the servo cases, dense surface samples (cropped) for thin shells."""
    if solid:
        from scipy.ndimage import binary_closing, binary_fill_holes
        S = m.sample(800000); l0 = S.min(0) - 1; shp = np.ceil((S.max(0) + 1 - l0) / 0.4).astype(int)
        g = np.zeros(shp, bool); i = np.floor((S - l0) / 0.4).astype(int); g[i[:, 0], i[:, 1], i[:, 2]] = True
        g = binary_fill_holes(binary_closing(g, iterations=1))
        return np.vstack([np.argwhere(g) * 0.4 + l0 + 0.2, S[::4]])
    keep, n = [], int(max(5000, m.area / 0.02))
    while n > 0:
        k = min(n, 500000); n -= k
        P = m.sample(k); keep.append(P[crop(P)])
    return np.vstack(keep)


def swept_grids(man):
    body = F.Body(man)
    J = {j["name"]: j for j in man["joints"]}
    shape = np.ceil((HI - LO) / V).astype(int)
    hw = ["R_upper_arm_shell_front_M3_nut", "R_upper_arm_shell_front_M3x8", "R_upper_arm_shell_rear_M3_nut",
          "R_upper_arm_shell_rear_M3x8", "R_yawcase_PA2x9p2_3", "R_yawcase_PA2x9p2_4", "upper__R_yaw_C047_case"]
    hw += [p["name"] for p in man["parts"] if p["name"].startswith("upper__R_yaw_case_")]
    out = {}
    for g, (names, jn, inv) in GROUPS.items():
        names = names or hw
        j = J[jn]; o = np.array(j["origin_world_mm"], float); ax = np.array(j["axis_world_neutral"], float)
        k = 0 if inv else 1   # crop along the rotation axis (x for pitch, y for roll)
        crop = lambda P, k=k: (P[:, k] > LO[k] - 1) & (P[:, k] < HI[k] + 1)
        P = np.vstack([part_points(body.mesh(n), "C047_case" in n, crop) for n in names])
        grid = np.zeros(shape, bool)
        for q in np.arange(j["range_deg"][0], j["range_deg"][1] + 0.01, 1.0):
            Q = (P - o) @ rotm(ax, -q if inv else q).T + o
            i = np.floor((Q - LO) / V).astype(int); ok = np.all((i >= 0) & (i < shape), 1); i = i[ok]
            grid[i[:, 0], i[:, 1], i[:, 2]] = True
        out[g] = grid
        print(f"  {g}: {len(P)} points swept over {jn} {j['range_deg']}", flush=True)
    return out


def axes(shape):
    return [LO[i] + V * (np.arange(shape[i]) + 0.5) for i in range(3)]


def vox(p):
    return tuple(np.floor((np.asarray(p) - LO) / V).astype(int))


def space():
    from scipy.ndimage import distance_transform_edt, label
    man = json.loads(MANIFEST.read_text())
    WORK.mkdir(parents=True, exist_ok=True); VIEWS.mkdir(parents=True, exist_ok=True)
    G = swept_grids(man)
    X, Y, Z = axes(G["shells"].shape)
    torso = G["shells"] | G["pitch_case"]
    roll = G["roll_case"] | G["cover"] | G["yaw_and_hardware"]
    rep = dict(manifest=str(MANIFEST.relative_to(ROOT)), voxel_mm=V, step_deg=1.0,
               note="pitch-link frame; torso swept over R_shoulder_pitch -60..150, roll link over R_shoulder_roll -15..95")

    # 1. occupancy of the owner's bar box
    m = np.ix_((X >= BAR_BOX["x"][0]) & (X <= BAR_BOX["x"][1]), (Y >= BAR_BOX["y"][0]) & (Y <= BAR_BOX["y"][1]),
               (Z >= BAR_BOX["z"][0]) & (Z <= BAR_BOX["z"][1]))
    rep["bar_box"] = dict(box_mm=BAR_BOX, occupied_pct={k: round(100 * float(G[k][m].mean()), 1) for k in G})

    # 2. x-extent of the sweeps by y band, z 340..375
    bands = []
    for y0, y1 in ((-14.1, -11.3), (-11.3, -9.8), (-9.8, 0), (0, 10), (10, 22.2), (22.2, 28)):
        mm = np.ix_(X < 104, (Y >= y0) & (Y < y1), (Z >= 340) & (Z <= 375)); Xs = X[X < 104]
        ws, wc = np.argwhere(G["shells"][mm]), np.argwhere(G["roll_case"][mm])
        bands.append(dict(y_mm=[y0, y1], shell_sweep_max_x=round(float(Xs[ws[:, 0]].max()), 1) if len(ws) else None,
                          roll_case_sweep_min_x=round(float(Xs[wc[:, 0]].min()), 1) if len(wc) else None))
    rep["x_extent_by_y"] = bands

    # 3. idler face: free radius about the roll axis before the swept roll-link parts, and which part
    rows = []
    for yv in (-14.5, -15.5, -16.5, -17.5, -18.5):
        j = int(np.argmin(abs(Y - yv))); row = {}
        for ang in range(60, 300, 15):
            a = np.radians(ang)
            for r in np.arange(2, 40, 0.25):
                i, k = vox((ROLL_O[0] + r * np.cos(a), 0, ROLL_O[2] + r * np.sin(a)))[0::2]
                if roll[i, j, k]:
                    row[ang] = (round(float(r), 2), "cover" if G["cover"][i, j, k] else "roll_case" if G["roll_case"][i, j, k] else "yaw/hardware")
                    break
        rows.append(dict(y_mm=yv, first_obstacle_r_mm_by_angle_deg=row))
    rep["idler_face_enclosure"] = rows

    # 4. connectivity horn side <-> idler side, by required free distance, with obstacle sets removed
    conn = {}
    for nm, O in (("all", torso | roll), ("without_cover", torso | G["roll_case"] | G["yaw_and_hardware"]),
                  ("without_torso_shells", G["pitch_case"] | roll),
                  ("without_cover_and_shells", G["pitch_case"] | G["roll_case"] | G["yaw_and_hardware"])):
        d = distance_transform_edt(~O) * V
        res = {}
        for need in (1.0, 1.5, 2.0, 2.5):
            lab, _ = label(d >= need)
            a, b = lab[vox(HORN_SEED)], lab[vox(IDLER_SEED)]
            res[str(need)] = bool(a and a == b)
        conn[nm] = res
        if nm == "without_cover":
            lab, _ = label(d >= 2.0); comp = lab == lab[vox(HORN_SEED)]
            cross = []
            for yv in (-11.0, -10.0):
                j = int(np.argmin(abs(Y - yv)))
                for z0, z1 in ((335, 350), (350, 365), (365, 380), (380, 390)):
                    w = np.argwhere(comp[:, j, (Z >= z0) & (Z < z1)])
                    cross.append(dict(y=yv, z=[z0, z1], x=[round(float(X[w[:, 0]].min()), 1), round(float(X[w[:, 0]].max()), 1)] if len(w) else None,
                                      voxels=int(len(w))))
            rep["crossing_without_cover_free_2mm"] = cross
    rep["horn_to_idler_connected"] = dict(seeds=dict(horn=HORN_SEED, idler=IDLER_SEED), by_free_distance_mm=conn)

    # 5. material the U would need from parts outside the brief (exact mesh containment on a 0.25 mm grid)
    body = F.Body(man); g = 0.25
    gx = np.arange(85, 91, g) + g / 2; gy = np.arange(-25, 45, g) + g / 2; gz = np.arange(320, 390, g) + g / 2
    P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
    r = np.hypot(P[:, 1] - PITCH_YZ[0], P[:, 2] - PITCH_YZ[1])
    sh = body.mesh("center_front_shell") + body.mesh("center_back_shell")
    rep["torso_shell_socket_relief_mm3"] = {f"r_{SOCKET_R}_to_{ro}": round(float(sh.contains(P[(r > SOCKET_R) & (r < ro) & (P[:, 0] >= 86)]).sum() * g ** 3), 0)
                                            for ro in (24, 26, 28)}
    cv = body.mesh("R_upper_arm_hard_cover")
    gx = np.arange(80, 125, g) + g / 2; gy = np.arange(-19, -12.9, g) + g / 2; gz = np.arange(335, 385, g) + g / 2
    P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
    dx, dz = P[:, 0] - ROLL_O[0], P[:, 2] - ROLL_O[2]; rr = np.hypot(dx, dz); th = np.degrees(np.arctan2(dz, dx)) % 360
    rep["cover_idler_arm_relief_mm3"] = dict(region="y -19..-13, r 8..21 about the roll axis, cover-frame angles 65..215 deg "
                                                    "(arm at 180 deg +-20 deg, roll -15..95)",
                                             volume=round(float(cv.contains(P[(rr > 8) & (rr < 21) & (th >= 65) & (th <= 215)]).sum() * g ** 3), 0),
                                             cover_volume=round(float(cv.volume), 0))
    (WORK / "space.json").write_text(json.dumps(rep, indent=1) + "\n")
    figure(G, X, Y, Z)
    print(json.dumps({k: rep[k] for k in ("bar_box", "x_extent_by_y", "horn_to_idler_connected", "torso_shell_socket_relief_mm3",
                                          "cover_idler_arm_relief_mm3", "crossing_without_cover_free_2mm")}, indent=1))


def figure(G, X, Y, Z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    col = dict(shells=(0.85, 0.2, 0.2), roll_case=(0.2, 0.35, 0.9), cover=(0.3, 0.8, 0.85), yaw_and_hardware=(0.6, 0.6, 0.6),
               pitch_case=(0.5, 0.2, 0.2))
    lbl = dict(shells="torso shells, swept by pitch -60..150", roll_case="roll case, swept by roll -15..95",
               cover="upper-arm cover, swept by roll", yaw_and_hardware="yaw case + hardware, swept by roll",
               pitch_case="pitch case")
    fig, axs = plt.subplots(1, 4, figsize=(26, 7.5))

    def paint(ax, sl, ext):
        img = np.ones(sl["shells"].shape + (3,))
        for k in ("yaw_and_hardware", "cover", "roll_case", "pitch_case", "shells"):
            img[sl[k]] = col[k]
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=ext, interpolation="nearest")

    for ax, yv, title in ((axs[0], -16.0, "y -16: idler-arm plane (idler face at y -13.3)"),
                          (axs[1], -12.0, "y -12: between roll-case face (y -11.3) and idler arm"),
                          (axs[2], -10.5, "y -10.5: the band the bar must cross")):
        j = int(np.argmin(abs(Y - yv)))
        paint(ax, {k: G[k][:, j, :] for k in G}, [X[0], X[-1], Z[0], Z[-1]])
        ax.add_patch(Rectangle((BAR_BOX["x"][0], BAR_BOX["z"][0]), 4, BAR_BOX["z"][1] - BAR_BOX["z"][0], fill=False, ec="k", lw=2, ls="--"))
        ax.text(BAR_BOX["x"][0] - 1, BAR_BOX["z"][1] + 1.5, "owner bar\nx 86-90", fontsize=9)
        ax.plot(ROLL_O[0], ROLL_O[2], "k+", ms=12); ax.add_patch(Circle((ROLL_O[0], ROLL_O[2]), 9.6, fill=False, ec="k", lw=1))
        ax.text(ROLL_O[0] + 1, ROLL_O[2] + 1, "roll axis\n(idler r 9.6)", fontsize=8)
        ax.set_title(title); ax.set_xlabel("x (mm, +R side)"); ax.set_ylabel("z (mm)"); ax.grid(alpha=0.3)
    ax = axs[3]; i = int(np.argmin(abs(X - 88.0)))
    paint(ax, {k: G[k][i, :, :] for k in G}, [Y[0], Y[-1], Z[0], Z[-1]])
    ax.add_patch(Circle(PITCH_YZ, SOCKET_R, fill=False, ec="y", lw=2)); ax.text(PITCH_YZ[0] - 8, PITCH_YZ[1] + SOCKET_R + 1, "torso socket r 18.8", fontsize=9)
    ax.add_patch(Rectangle((-18.0, BAR_BOX["z"][0]), 45.5, BAR_BOX["z"][1] - BAR_BOX["z"][0], fill=False, ec="k", lw=2, ls="--"))
    ax.text(-18, BAR_BOX["z"][0] - 3.5, "owner bar y -18..27.5 (idler arm .. horn arm)", fontsize=9)
    ax.plot(*PITCH_YZ, "y+", ms=12); ax.plot(ROLL_O[1], ROLL_O[2], "k+", ms=12)
    ax.set_title("x 88: section through the owner bar (y-z)"); ax.set_xlabel("y (mm, -face)"); ax.set_ylabel("z (mm)"); ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col[k], label=lbl[k]) for k in ("shells", "pitch_case", "roll_case", "cover", "yaw_and_hardware")],
               loc="lower center", ncol=5, fontsize=10)
    fig.suptitle("Shoulder roll clevis: space for stage 1 in the pitch-link frame (white = free over both full joint ranges)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    f = VIEWS / "shoulder-clevis-space.png"
    fig.savefig(f, dpi=80)
    print(f"wrote {f.relative_to(ROOT)}")


# ======================================================================================================================
# build / manifest: the two U's as the owner drew them (2026-11-20, Idle T-2e49mv): "Design the U for the blue and green
# as defined right now and then we look at the actual parts and fix collisions from there." Collisions with the torso
# shells and the upper-arm cover are measured afterwards, not designed around.
#
#   Stage 1 (blue, R_shoulder_pitch_to_roll_combined, pitch link): today's pitch interface (x 82.625..86, 4 sleeves, kept
#     from today's B-rep), a bar along y at x 86..90, and two arms running +x on both roll-servo faces: horn arm on the
#     roll main horn (+y, 4 x M3 horn+y), idler arm on a new roll idler (-y, 4 x M3 idler-y).
#   Stage 2 (green, R_shoulder_roll_to_yaw_combined, roll link), rebuilt from scratch (owner M-s9tnc9: simpler): a U
#     around the roll case's far (+x) end: two arms on the case's y faces (case frame, registered STS3250 transform)
#     clamping it with the far holes only (case+y.3/.4, case-y.3/.4), and as the U's bar one plain plate across the far
#     end face carrying today's yaw case clamp (4 PA2 sites, sleeves, spacers). No ribs / gussets / fillers; the
#     upper-arm cover posts are dropped (cover redesigned with the shell).
#   Roll idler: STS3250 STEP member "idler" (solid 12) placed with the registered roll case transform, on the pitch link.
#
#   .venv/bin/python -B robot/body/arm/shoulder_clevis.py build      (B-rep -> STEP / STL, cad-manifest.json)
#   .venv/bin/python -B robot/body/arm/shoulder_clevis.py manifest   (candidate manifest, screws from screw_plan)
#   .venv/bin/python -B robot/body/arm/shoulder_clevis.py collide    (standing overlaps, mm3 per pair)
#   .venv/bin/python -B robot/body/arm/shoulder_clevis.py views      (the two owner images)
# L side = mirror x -> -x of the R parts.
# ======================================================================================================================
OUT = ROOT / "artifacts/robot/body/arm/shoulder-clevis"
PARTS_DIR = OUT / "parts"
CANDIDATE = OUT / "assembly-parts-shoulder-u.json"
STS_STEP = ROOT / "artifacts/robot/actuators/interfaces/sources/feetech_sts3250.step"
TODAY = "artifacts/cove50-strength/complete-fdm-body-20261106/parts"
S1, S2 = "R_shoulder_pitch_to_roll_combined", "R_shoulder_roll_to_yaw_combined"
IDLER = "upper__R_roll_C047_rear_idler"
ROLL_XZ = (106.0, 360.0)

# Stage 1 (body frame, mm)
PITCH_KEEP_X = 86.0                    # today's part kept for x <= this: the pitch horn interface, unchanged
BAR_X, BAR_Z = (86.0, 90.0), (346.0, 370.0)
HORN_FACE_Y = 23.68                    # outer face of the roll horn mesh (the defined hole opening is 23.28)
HORN_ARM_T = 4.6                       # horn arm 23.68 .. 28.28: stack 5.0 from the hole opening -> M3x8 at reference engagement
IDLER_FACE_Y = -13.33                  # idler outer face (official STEP, registered case transform)
IDLER_ARM_T = 3.4                      # idler arm -13.33 .. -16.73: stack 3.4 -> M3x6 at reference engagement
HUB_R = 12.0                           # arm hub about the roll axis (bolt circle r 7 + sleeve r 2.25 + 2.75 wall)
HORN_CENTRE_R = 3.2                    # rocker screw head (dia 5.7) inside the horn arm
IDLER_CENTRE_R = 3.6                   # idler shaft stub (r <= 3.0, to y -14.13): 0.6 mm radial clearance
PITCH_BORE_R, PITCH_CENTRE_R = 3.35, 3.1   # today's pitch-screw head counterbores / rocker-screw head pocket
PITCH_YZ_C = (9.0, 352.0)              # pitch axis (along x); the 4 pitch sleeves sit on r 7 about it (x 82.625 .. 85.0)
PITCH_SLEEVE_X = (82.625, 85.0)        # today's pitch sleeves: entry end on the horn face, exit plane = counterbore floor
PITCH_HUB_R = 13.5                     # solid hub disc about the pitch axis, x 82.625 .. bar end: counterbore r 3.35 at r 7
                                       # + 3.15 wall; fills the kept B-rep's breakout at sleeve 2 (y 16) and its cheek pocket
                                       # (inner arc r 12.7 .. 13.0 at x > 85.4, which left a 0.04 mm gap at r 12.1)
PITCH_ROCKER_TOP_X = 84.6              # rocker-screw head pocket r 3.1 up to here (head x 82.625 .. 84.275); above it only a
PITCH_DRIVER_R = 1.6                   # hex-key channel (blade dia 3.0) to the bar face: 2.05 mm wall to each counterbore
                                       # (the r 3.1 hole carried through the bar left 0.55 mm walls)
M3_SLEEVE = (2.25, 1.60)               # 304 sleeve 4.5 x 3.2 (r outer, r bore); cut = outer radius, as the hip roll-flip
M2_SLEEVE = (1.80, 1.10)               # 304 sleeve 3.6 x 2.2 for the M2 case holes (as the hip roll-case sleeves; a 3.2 bore
                                       # leaves an M2 button head (dk 3.5) 0.15 mm of bearing ring)

# Stage 2 (canonical STS3250 case frame: output axis along +y through (x -25.5, z 0); case x -35.61..9.61, z +-12.36;
# +y face 6.4 at the strips |z| > 10, raised panel 7.5; -y face -25.6 at the strips, raised panel -27.5 over x -9..4.5)
G_PY = (7.6, 14.6)                     # +y arm: 0.1 mm above the raised panel, 7.0 thick (outward): stack 8.4 -> M2x12
G_NY = (-33.6, -27.6)                  # -y arm: 0.1 mm off the raised panel, 6.0 thick (outward): stack 8.2 -> M2x12
G_ROOT_R = (1.2, 1.5)                  # arm-to-plate root fillets inside the U (+y, -y), along the case's far end edges:
                                       # +y clears the case's end chamfer by ~0.4 mm, -y the case face (y -25.6 there) by 0.5
G_X = (-1.5, 15.0)                     # arm extent along the case (far holes at x 3.49: sleeve r 1.8 + 3.2 wall toward the
                                       # horn; far end face at 9.61: the arms run 5.4 mm past it, their corners buried in the plate)
G_Z = (-14.6, 14.66)                   # arm extent across the case (far holes at z +-10.25: sleeve r 1.8 + 2.55 wall); the +z
                                       # edge is slanted (see G_BORE_CLEAR), G_Z[1] only bounds the root fillet blocks
G_ARM_R = 3.0                          # arm corner radius (2D)
G_BORE_CLEAR = 0.5                     # the arms' slanted +z edge: clearance to the head / driver bore at (v +10.25, w 32.75)
# No separate bar on the case side face z +12.36 (the first green had one at z 12.66 .. 16.66): there it occupies yaw-local
# v ~7.6 .. 11.6 at n 20.6 .. 24.5, where today's PA2x9.2 heads of the two outer yaw sites sit (v 8.35 .. 12.15); the
# plate under the case's far end face is the U's bar.
G_END_GAP = 0.3                        # plate top = the plane 0.3 mm below the case far end face (x 9.61)
G_END_PLANE = 9.61 + G_END_GAP
# Yaw clamp: today's interface (assemble_complete_fdm_body stage 2). Yaw-local frame: origin YAW_P, v = YAW_V, w = +y (from
# YAW_P), n = YAW_N. 4 PA2 sites at v +-10.25, w 8.30 / 32.75; sleeves dia 3.5 x 2.4 at n 17.575 .. 20.575 (today's STEP),
# spacers n 16.57 .. 17.58 on the yaw case tabs, washers / heads on n 20.575. Yaw case top (measured, mesh): n <= 16.6 at the
# tabs, a raised pad v +-9.13, w 18.11 .. 29.59 to n 18.48, an axis boss r 3.0 at (0, 0) to n 19.18.
YAW_P = np.array([127.0, -10.0, 306.0])
YAW_N = np.array([-0.1643989873053573, 0.0, 0.9863939238321437])
YAW_V = np.array([0.9863939238321437, 0.0, 0.1643989873053573])
YAW_SITES = [(u, w) for u in (-10.25, 10.25) for w in (8.30, 32.75)]
PLATE_N = (17.575, 30.0)               # plate: bottom = today's saddle bottom (sleeve / spacer interface); top = the end-face
                                       # plane (2.7 mm thick at the -v end .. 7.9 at +v; the old cap at n 22 is gone)
PLATE_V = (-24.0, 14.0)                # +v: 1.5 mm wall around the head pockets at v 10.25; -v: run out past the arms and
                                       # cut flush with their -z face (case z G_Z[0]), so the plate spans the whole arm
PLATE_R = 3.0                          # plate corner radius (2D)
POCKET = dict(v=(-9.55, 9.55), w=(17.8, 29.9), n_top=19.0, r=1.5)   # clears the yaw case's raised pad by 0.3..0.5 mm
AXIS_HOLE_R = 3.45                     # today's centre clearance over the yaw case axis boss (r 3.0)
SLEEVE_BORE_R = 1.79                   # today's bore for the 3.5 x 2.4 yaw sleeves
HEAD_POCKET_R = 2.25                   # head counterbore r (PA2x9.2 head dk 3.5) from n 20.555 as today, run up to n 60 as
                                       # the driver channel too (PH0 blade dia 3.0): one bore, no steps; under the case it
                                       # leaves through the plate top, under the +y arm it runs through the arm and breaks
                                       # its inner face at 61 deg (a dia 3.2 channel would leave a 47 deg wedge)


def _occ():
    from build123d import Solid, Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    return locals()


def occ_box(x0, x1, y0, y1, z0, z1):
    o = _occ()
    return o["Solid"](o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](min(x0, x1), min(y0, y1), min(z0, z1)),
                                               o["gp_Pnt"](max(x0, x1), max(y0, y1), max(z0, z1))).Shape())


def occ_cyl(p0, p1, r):
    o = _occ()
    v = np.asarray(p1, float) - np.asarray(p0, float); L = float(np.linalg.norm(v)); d = v / L
    ax = o["gp_Ax2"](o["gp_Pnt"](*map(float, p0)), o["gp_Dir"](*map(float, d)))
    return o["Solid"](o["BRepPrimAPI_MakeCylinder"](ax, float(r), L).Shape())


def _bool(kind, a, others):
    o = _occ()
    op = {"fuse": o["BRepAlgoAPI_Fuse"], "cut": o["BRepAlgoAPI_Cut"], "common": o["BRepAlgoAPI_Common"]}[kind]
    cur = a.wrapped
    for s in others:
        b = op(cur, s.wrapped); b.Build(); cur = b.Shape()
    return o["Compound"](cur)


def largest(sh):
    return max(sh.solids(), key=lambda s: s.volume)


def transformed(shape, T):
    """shape moved by the 4x4 rigid transform T (rotation orthonormalised)."""
    from build123d import Compound
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    U, _, Vt = np.linalg.svd(T[:3, :3]); R = U @ Vt; t = T[:3, 3]
    tr = gp_Trsf()
    tr.SetValues(*[float(v) for v in (R[0, 0], R[0, 1], R[0, 2], t[0], R[1, 0], R[1, 1], R[1, 2], t[1], R[2, 0], R[2, 1], R[2, 2], t[2])])
    return largest(Compound(BRepBuilderAPI_Transform(shape.wrapped, tr, True).Shape()))


def mirror_x(sh):
    from build123d import Plane
    return sh.mirror(Plane.YZ)


def sleeve(p_in, p_out, ro, ri):
    """Steel sleeve from p_in to p_out (flush with both faces of the part)."""
    a, b = np.asarray(p_in, float), np.asarray(p_out, float); d = (b - a) / np.linalg.norm(b - a)
    return largest(_bool("cut", occ_cyl(a, b, ro), [occ_cyl(a - 0.5 * d, b + 0.5 * d, ri)]))


def _common_mod():
    sys.path.insert(0, str(ROOT / "robot/body/leg"))
    import common as C   # fillet_2d_face, extrude_xy_to_y, make_captive_nut_pocket (read-only use)
    return C


def today_step(name):
    from build123d import import_step
    sys.path.insert(0, str(ROOT))
    from infra.storage.blob_store import ensure
    return import_step(str(ensure(ROOT / TODAY / f"{name}.step")))


def registered(man=None):
    """Registered STS3250 instances of the right shoulder (robot/body/fastening/interfaces.py): roll case transform and
    the defined holes of the roll case and roll horn (body frame); idler holes = the definition under the case transform."""
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    import interfaces as I
    man = man or json.loads(MANIFEST.read_text())
    need = {"upper__R_roll_C047_case", "upper__R_roll_C047_main_horn"}
    sub = dict(man, parts=[p for p in man["parts"] if p["name"] in need or p.get("motor")])
    iface = I.instances(F.Body(sub), sub)
    row = next(r for r in iface["instances"] if r["part"] == "upper__R_roll_C047_case")
    T = np.array(row["transform"], float); U, _, Vt = np.linalg.svd(T[:3, :3]); T[:3, :3] = U @ Vt
    holes = {h["label"]: h for h in iface["holes"] if h["part"] == "upper__R_roll_C047_case"}
    holes.update({h["label"]: h for h in iface["holes"] if h["part"] == "upper__R_roll_C047_main_horn"})
    defs = json.loads(I.DEFS.read_text())["motors"]["STS3250"]
    for h in defs["mount_holes"]:
        if h["member"] == "idler":
            holes[h["hole_id"]] = dict(label=h["hole_id"], opening=T[:3, :3] @ np.array(h["opening_mm"]) + T[:3, 3],
                                       axis_out=T[:3, :3] @ np.array(h["axis_out"]), thread=h["thread"])
    return T, holes, {r["part"]: {k: r[k] for k in ("fit_rms_mm", "modelled_with", "hole_snap")} for r in iface["instances"] if r["part"] in need}


def build_idler(T):
    from build123d import import_step
    return transformed(import_step(str(STS_STEP)).solids()[12], T)


def build_stage1(holes):
    """Blue U (pitch link): today's pitch interface + bar along y at x 86..90 + horn / idler arms, and a solid pitch hub
    disc (r PITCH_HUB_R about the pitch axis, horn face .. bar face) around today's 4 pitch sleeve sites."""
    from build123d import Circle, Polygon, Pos
    C = _common_mod()
    rx, rz = ROLL_XZ
    hy0, hy1 = HORN_FACE_Y, HORN_FACE_Y + HORN_ARM_T
    iy1, iy0 = IDLER_FACE_Y, IDLER_FACE_Y - IDLER_ARM_T
    # 1. pitch interface: today's B-rep for x <= 86 (pitch cheek, bosses, ring, the 4 sleeve bores, counterbores)
    keep = largest(_bool("common", largest(today_step(S1)), [occ_box(80.0, PITCH_KEEP_X, -30, 40, 330, 380)]))
    # 2. bar along y, x 86..90, from the idler arm's outer face to the horn arm's outer face
    bar = occ_box(BAR_X[0], BAR_X[1], iy0, hy1, BAR_Z[0], BAR_Z[1])
    # 3. arms (x-z profile): hub r HUB_R about the roll axis + a neck back to the bar
    neck = Polygon((BAR_X[0], BAR_Z[0]), (rx, rz - HUB_R), (rx, rz + HUB_R), (BAR_X[0], BAR_Z[1]), align=None)
    prof = C.fillet_2d_face((neck + Pos(rx, rz) * Circle(HUB_R)).clean().faces()[0], 2.0, r_inner=4.0,
                            skip_xy=[(BAR_X[0], BAR_Z[0]), (BAR_X[0], BAR_Z[1])])
    arm_h = C.extrude_xy_to_y(prof, hy0, hy1)
    arm_i = C.extrude_xy_to_y(prof, iy0, iy1)
    # 4. pitch hub: a solid disc about the pitch axis from the horn face to the bar's outer face (fills the kept B-rep's
    # breakout at sleeve 2 and its open counter faces); the sleeve bores, counterbores and centre are cut again below
    py, pz = PITCH_YZ_C
    hub = occ_cyl((PITCH_SLEEVE_X[0], py, pz), (BAR_X[1], py, pz), PITCH_HUB_R)
    body = largest(_bool("fuse", keep, [bar, arm_h, arm_i, hub]))
    cut = [occ_cyl((rx, hy0 - 1, rz), (rx, hy1 + 1, rz), HORN_CENTRE_R),
           occ_cyl((rx, iy0 - 1, rz), (rx, iy1 + 1, rz), IDLER_CENTRE_R)]
    # today's pitch sleeve bores (r = sleeve OD / 2 as every other sleeve here) and head counterbores from the sleeve exit
    # plane through the bar (driver access to the pitch horn screws; the heads pass through them)
    for (u, v) in ((-7.0, 0.0), (0.0, 7.0), (7.0, 0.0), (0.0, -7.0)):
        cut.append(occ_cyl((PITCH_SLEEVE_X[0] - 1.0, py + u, pz + v), (PITCH_SLEEVE_X[1], py + u, pz + v), M3_SLEEVE[0]))
        cut.append(occ_cyl((PITCH_SLEEVE_X[1], py + u, pz + v), (BAR_X[1] + 1.0, py + u, pz + v), PITCH_BORE_R))
    # rocker-screw head pocket (the horn is screwed to the servo before the U goes on), then a hex-key channel
    cut.append(occ_cyl((80.0, py, pz), (PITCH_ROCKER_TOP_X, py, pz), PITCH_CENTRE_R))
    cut.append(occ_cyl((PITCH_ROCKER_TOP_X, py, pz), (BAR_X[1] + 1.0, py, pz), PITCH_DRIVER_R))
    sleeves, sites = [], []
    for k in range(1, 5):
        o = holes[f"STS3250.horn+y.{k}"]["opening"]
        a, b = (o[0], hy0, o[2]), (o[0], hy1, o[2])
        cut.append(occ_cyl((o[0], hy0 - 1.0, o[2]), (o[0], hy1 + 1.0, o[2]), M3_SLEEVE[0]))
        sleeves.append(sleeve(a, b, *M3_SLEEVE)); sites.append(dict(hole=f"STS3250.horn+y.{k}", clamp=a, head_face=b))
        o = holes[f"STS3250.idler-y.{k}"]["opening"]
        a, b = (o[0], iy1, o[2]), (o[0], iy0, o[2])
        cut.append(occ_cyl((o[0], iy1 + 1.0, o[2]), (o[0], iy0 - 1.0, o[2]), M3_SLEEVE[0]))
        sleeves.append(sleeve(a, b, *M3_SLEEVE)); sites.append(dict(hole=f"STS3250.idler-y.{k}", clamp=a, head_face=b))
    part = largest(_bool("cut", body, cut))
    # today's 4 pitch sleeves (x 82.625 .. 85.0), unchanged
    pitch = [s for s in today_step(S1 + "_steel_sleeves").solids() if s.bounding_box().max.X < 86.0]
    return part, pitch + sleeves, sites


def yaw_pt(v, w, n):
    """Yaw-local (v, w, n) -> body frame."""
    return YAW_P + v * YAW_V + w * np.array([0.0, 1.0, 0.0]) + n * YAW_N


def build_stage2(T, holes):
    """Green U (roll link), from scratch (owner 2026-11-20, M-s9tnc9: simpler): a U around the roll case's far end: two
    arms on the case's y faces (built in the case frame, placed with the registered case transform) and, as its bar, one
    plain plate across the far end face carrying today's yaw case clamp (built in the yaw-local frame). The plate fills
    the space between the yaw case top and the roll case's far end face (top = the plane 0.3 mm below the end face) and
    spans the arms' full width along the case (cut flush with their -z face). Root fillets join the arms' inner faces to
    the plate top along the case's far end edges. Cuts: the 4 far-hole M2 sleeves, one pocket over the yaw case's raised
    pad, the axis-boss hole, and per yaw site its sleeve bore and one head / driver bore (opened toward the case above the
    plate top at the two sites under the +y arm).
    Print direction intended: plate bottom on the bed (up = YAW_N, the yaw-local normal, 9.5 deg off +z)."""
    from build123d import Polygon, Pos, RectangleRounded
    C = _common_mod()
    Ti = np.linalg.inv(T)
    to_c = lambda p: Ti[:3, :3] @ np.asarray(p, float) + Ti[:3, 3]
    M_yaw = np.eye(4); M_yaw[:3, 0], M_yaw[:3, 1], M_yaw[:3, 2], M_yaw[:3, 3] = YAW_V, (0.0, 1.0, 0.0), YAW_N, YAW_P

    def rrect(a0, a1, b0, b1, r):
        return (Pos(0.5 * (a0 + a1), 0.5 * (b0 + b1)) * RectangleRounded(a1 - a0, b1 - b0, r)).faces()[0]

    case_side = transformed(occ_box(-100.0, G_END_PLANE, -100.0, 100.0, -100.0, 100.0), T)     # case side of the plate top
    # 1. arms, case frame: x-z outline extruded across y. The +z edge runs parallel to the yaw axis, 0.5 mm clear of the
    # head / driver bore at (v +10.25, w 32.75): that bore leaves the +y arm at a slant otherwise (a 0.05 mm knife edge
    # along the old straight edge z 14.66); the edge stays 2.2 mm off the far-hole sleeve at z +10.25
    bore = [to_c(yaw_pt(10.25, 32.75, nn))[[0, 2]] for nn in (PLATE_N[0], 60.0)]
    d = (bore[1] - bore[0]) / np.linalg.norm(bore[1] - bore[0]); nrm = np.array([d[1], -d[0]])
    nrm = nrm if nrm[1] < 0 else -nrm                                      # toward -z
    off = bore[0] + (HEAD_POCKET_R + G_BORE_CLEAR) * nrm
    z_edge = lambda x: off[1] + (x - off[0]) * d[1] / d[0]
    arm_face = C.fillet_2d_face(Polygon((G_X[0], G_Z[0]), (G_X[1], G_Z[0]), (G_X[1], z_edge(G_X[1])), (G_X[0], z_edge(G_X[0])),
                                        align=None).faces()[0], G_ARM_R)
    arms = [transformed(C.extrude_xy_to_y(arm_face, *ys), T) for ys in (G_PY, G_NY)]
    arm_xz = transformed(C.extrude_xy_to_y(arm_face, -100.0, 100.0), T)     # the outline as a prism across y
    # 2. plate (the U's bar), yaw-local frame: rounded rectangle in v-w, 0.5 mm inside the arms' outer faces in y (flush,
    # the case transform's 1e-4 rad tilt left sliver faces along the joint), extruded along n; top = the end-face plane,
    # -v end flush with the arms' -z face
    w_lo, w_hi = T[1, 3] + G_NY[0] - YAW_P[1] + 0.5, T[1, 3] + G_PY[1] - YAW_P[1] - 0.5
    plate = transformed(C.extrude_xy_to_z(rrect(*PLATE_V, w_lo, w_hi, PLATE_R), *PLATE_N), M_yaw)
    minus_z = transformed(occ_box(-100.0, 100.0, -100.0, 100.0, -100.0, G_Z[0]), T)
    plate = largest(_bool("cut", plate, [case_side, minus_z]))
    below = transformed(occ_box(-100.0, 100.0, -100.0, 100.0, -60.0, PLATE_N[0]), M_yaw)       # bed side of the plate bottom
    # 3. root fillets (case frame): the corner between each arm's inner face and the plate top, along z over the arm
    roots = []
    for y_face, s, r in ((G_PY[0], 1.0, G_ROOT_R[0]), (G_NY[1], -1.0, G_ROOT_R[1])):
        xc, yc = G_END_PLANE - r, y_face - s * r
        blk = occ_box(xc, G_END_PLANE, yc, y_face, G_Z[0], G_Z[1])
        roots.append(largest(_bool("common", transformed(largest(_bool("cut", blk, [occ_cyl((xc, yc, G_Z[0] - 1.0), (xc, yc, G_Z[1] + 1.0), r)])), T),
                                   [arm_xz])))
    body = largest(_bool("fuse", plate, [largest(_bool("cut", a, [below])) for a in arms] + roots))
    # 4. cutters
    cut = [transformed(C.extrude_xy_to_z(rrect(*POCKET["v"], *POCKET["w"], POCKET["r"]), PLATE_N[0] - 1.0, POCKET["n_top"]), M_yaw),
           occ_cyl(yaw_pt(0, 0, PLATE_N[0] - 1.0), yaw_pt(0, 0, PLATE_N[1] + 5.0), AXIS_HOLE_R)]
    sleeves, sites = [], []
    R, t = T[:3, :3], T[:3, 3]
    for lab, (fa, fb) in (("case+y.3", G_PY), ("case+y.4", G_PY), ("case-y.3", G_NY[::-1]), ("case-y.4", G_NY[::-1])):
        o = to_c(holes[f"STS3250.{lab}"]["opening"])
        a_c, b_c = np.array([o[0], fa, o[2]]), np.array([o[0], fb, o[2]])
        d = (b_c - a_c) / np.linalg.norm(b_c - a_c)
        a, b = R @ a_c + t, R @ b_c + t
        cut.append(occ_cyl(R @ (a_c - d) + t, R @ (b_c + d) + t, M2_SLEEVE[0]))
        sleeves.append(sleeve(a, b, *M2_SLEEVE)); sites.append(dict(hole=f"STS3250.{lab}", clamp=a, head_face=b))
    for u, w in YAW_SITES:   # today's sleeve bore (n 16.575 .. 20.675); one head / driver bore from n 20.555 (as today) up
        tool = occ_cyl(yaw_pt(u, w, 20.555), yaw_pt(u, w, 60.0), HEAD_POCKET_R)
        if w > 20.0 and u < 0:
            # under the +y arm (v -10.25) the bore runs 1.08 mm inside the arm's inner face and through its root fillet:
            # opened into a slot toward the case (-w), above the plate top only, so no knife edges are left along them
            # (the v +10.25 bore passes outside the arm's slanted +z edge)
            slot = largest(_bool("common", transformed(occ_box(u - HEAD_POCKET_R, u + HEAD_POCKET_R, w - 15.0, w, 15.0, 60.0), M_yaw),
                                 [case_side]))
            tool = largest(_bool("fuse", tool, [slot]))
        cut += [occ_cyl(yaw_pt(u, w, 16.575), yaw_pt(u, w, 20.675), SLEEVE_BORE_R), tool]
    part = largest(_bool("cut", body, cut))
    yaw = [s for s in today_step(S2 + "_steel_sleeves").solids() if s.center().Y > -10.0 and s.center().Z < 330.0]
    return part, yaw + sleeves, sites


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def _export(name, shapes):
    from build123d import Compound, export_step, export_stl
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    comp = Compound(children=list(shapes))
    st, sl = PARTS_DIR / f"{name}.step", PARTS_DIR / f"{name}.stl"
    export_step(comp, str(st))
    import trimesh
    for tol, ang in ((0.02, 0.08), (0.035, 0.09)):   # first closed tessellation
        export_stl(comp, str(sl), tolerance=tol, angular_tolerance=ang)
        if trimesh.load(str(sl), force="mesh", process=True).is_watertight:
            break
    bb = comp.bounding_box()
    return dict(name=name, native_step=_rel(st), native_step_sha256=_sha(st), mesh=_rel(sl), mesh_sha256=_sha(sl),
                bounds_mm=[[round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)], [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]],
                volume_mm3=round(float(sum(s.volume for s in shapes)), 2), solids=len(shapes),
                valid=bool(all(s.is_valid for s in shapes)))


def _density(part):
    data = ROOT / "robot/body/checks/data"
    mat = json.loads((data / "part_settings.json").read_text())[part]["process"]["material"]
    return mat, json.loads((data / "materials.json").read_text())[mat]["density_g_cm3"]


def sections():
    """Wall / bar sections of the built parts (mm), from the constants."""
    return dict(
        stage1=dict(bar=dict(x=list(BAR_X), z=list(BAR_Z), section_mm=[BAR_X[1] - BAR_X[0], BAR_Z[1] - BAR_Z[0]],
                             y=[round(IDLER_FACE_Y - IDLER_ARM_T, 2), round(HORN_FACE_Y + HORN_ARM_T, 2)]),
                    horn_arm=dict(y=[HORN_FACE_Y, round(HORN_FACE_Y + HORN_ARM_T, 2)], thickness=HORN_ARM_T, hub_r=HUB_R),
                    idler_arm=dict(y=[round(IDLER_FACE_Y - IDLER_ARM_T, 2), IDLER_FACE_Y], thickness=IDLER_ARM_T, hub_r=HUB_R),
                    neck_height_at_bar=BAR_Z[1] - BAR_Z[0],
                    pitch_hub=dict(axis_yz=list(PITCH_YZ_C), x=[PITCH_SLEEVE_X[0], BAR_X[1]], r=PITCH_HUB_R,
                                   sleeve_bore_r=M3_SLEEVE[0], counterbore_r=PITCH_BORE_R, counterbore_from_x=PITCH_SLEEVE_X[1],
                                   rocker_pocket=dict(r=PITCH_CENTRE_R, to_x=PITCH_ROCKER_TOP_X), driver_channel_r=PITCH_DRIVER_R)),
        stage2=dict(frame="U in the STS3250 case frame (registered roll case transform), plate in the yaw-local frame "
                          "(origin YAW_P, v, w = +y from YAW_P, n = YAW_N)",
                    py_arm=dict(case_y=list(G_PY), thickness=round(G_PY[1] - G_PY[0], 2)),
                    ny_arm=dict(case_y=list(G_NY), thickness=round(G_NY[1] - G_NY[0], 2)),
                    arm_extent_case_x=list(G_X), arm_extent_case_z=list(G_Z), arm_corner_r=G_ARM_R,
                    root_fillet_r=dict(py=G_ROOT_R[0], ny=G_ROOT_R[1]),
                    bar="the plate (no separate bar on the case side face: it would sit on the outer PA2x9.2 heads)",
                    plate=dict(v=list(PLATE_V), n=list(PLATE_N), corner_r=PLATE_R,
                               top=f"plane {G_END_GAP} mm below the roll case end face (case x {G_END_PLANE:.2f})",
                               minus_z_end=f"flush with the arms' -z face (case z {G_Z[0]})",
                               thickness_mm="2.7 at the -v end (under the case end) .. 7.9 at v +14",
                               pocket=POCKET, axis_hole_r=AXIS_HOLE_R),
                    yaw_sites=dict(v_w=YAW_SITES, sleeve_bore_r=SLEEVE_BORE_R, head_and_driver_bore_r=HEAD_POCKET_R,
                                   head_and_driver_bore_n=[20.555, 60.0]),
                    print_up="yaw-local n (plate bottom on the bed)"))


def build():
    import time
    t0 = time.time()
    T, holes, fits = registered()
    print(f"registered ({time.time() - t0:.1f} s): {fits}", flush=True)
    s1, s1_slv, s1_sites = build_stage1(holes)
    print(f"stage 1 built ({time.time() - t0:.1f} s): {s1.volume:.0f} mm3, valid {s1.is_valid}", flush=True)
    s2, s2_slv, s2_sites = build_stage2(T, holes)
    print(f"stage 2 built ({time.time() - t0:.1f} s): {s2.volume:.0f} mm3, valid {s2.is_valid}", flush=True)
    idl = build_idler(T)
    rows = {}
    for side, f in (("R_", lambda s: s), ("L_", mirror_x)):
        for base, sol, slv in ((S1, s1, s1_slv), (S2, s2, s2_slv)):
            nm = base.replace("R_", side, 1)
            r = _export(nm, [f(sol)]); mat, rho = _density(nm)
            r.update(material=mat, density_g_cm3=rho, mass_g=round(r["volume_mm3"] * rho * 1e-3, 2))
            rows[nm] = r
            rs = _export(nm + "_steel_sleeves", [f(s) for s in slv])
            rs.update(material="304 stainless", mass_g=round(rs["volume_mm3"] * 7.93e-3, 2)); rows[nm + "_steel_sleeves"] = rs
        nm = IDLER.replace("upper__R_", "upper__" + side)
        rows[nm] = _export(nm, [f(idl)])
    sites = {"stage1": s1_sites, "stage2": s2_sites}
    for v in sites.values():
        for s in v:
            s["clamp"] = np.round(s["clamp"], 3).tolist(); s["head_face"] = np.round(s["head_face"], 3).tolist()
    rep = dict(owner="2026-11-20: build the blue and green U as drawn; collisions measured afterwards (Idle T-2e49mv)",
               frame="body frame of the selected manifest (+x R side, -y face, +z up); L = mirror x",
               selected_manifest=str(MANIFEST.relative_to(ROOT)), registered_instances=fits,
               roll_case_transform=np.round(T, 6).tolist(), sections_mm=sections(), sleeve_sites_R=sites,
               sleeves=dict(M3="304, dia 4.5 x 3.2 bore (r 2.25 / 1.60), flush with both faces",
                            M2="304, dia 3.6 x 2.2 bore (r 1.80 / 1.10), flush with both faces",
                            kept="today's 4 pitch sleeves (dia 4.5 x 3.3, x 82.625..85.0) and 4 yaw-saddle sleeves (dia 3.5 x 2.4)"),
               parts=rows, seconds=round(time.time() - t0, 1),
               source=dict(path=_rel(__file__), sha256=_sha(__file__)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cad-manifest.json").write_text(json.dumps(rep, indent=1) + "\n")
    for k, r in rows.items():
        if k.startswith("R_") or k.startswith("upper__R"):
            print(f"  {k:52s} {r['volume_mm3']:9.1f} mm3  {r.get('mass_g', '')} g  solids {r['solids']} valid {r['valid']}  {r['bounds_mm']}")
    print(f"wrote {(OUT / 'cad-manifest.json').relative_to(ROOT)} ({time.time() - t0:.1f} s)")


# ---------------------------------------------------------------------------------------------- candidate manifest
SPECS = {   # screw_plan.plan_connector specs (connector_screws.json format; not added to that file)
    "shoulder_u_stage1": {"clamped": ["{S}shoulder_pitch_to_roll_combined", "{S}shoulder_pitch_to_roll_combined_steel_sleeves"],
                          "groups": {"roll_horn": {"standard_part": "upper__{S}roll_C047_main_horn", "head": "ISO 7380-1",
                                                   "holes": [f"STS3250.horn+y.{k}" for k in range(1, 5)],
                                                   "name": "{S}shoulder_u_roll_horn_screw_{i}", "owning_link": "{S}shoulder_pitch_link"},
                                     "roll_idler": {"standard_part": "upper__{S}roll_C047_rear_idler", "head": "ISO 7380-1",
                                                    "holes": [f"STS3250.idler-y.{k}" for k in range(1, 5)],
                                                    "name": "{S}shoulder_u_roll_idler_screw_{i}", "owning_link": "{S}shoulder_pitch_link"}}},
    "shoulder_u_stage2": {"clamped": ["{S}shoulder_roll_to_yaw_combined", "{S}shoulder_roll_to_yaw_combined_steel_sleeves"],
                          "groups": {"roll_case_py": {"standard_part": "upper__{S}roll_C047_case", "head": "ISO 7380-1",
                                                      "holes": ["STS3250.case+y.3", "STS3250.case+y.4"],
                                                      "name": "{S}shoulder_u_roll_case_py_screw_{i}", "owning_link": "{S}shoulder_roll_link"},
                                     "roll_case_ny": {"standard_part": "upper__{S}roll_C047_case", "head": "ISO 7380-1",
                                                      "holes": ["STS3250.case-y.3", "STS3250.case-y.4"],
                                                      "name": "{S}shoulder_u_roll_case_ny_screw_{i}", "owning_link": "{S}shoulder_roll_link"}}},
}
REMOVED = ([f"upper__{{S}}roll_case_{k}{i}" for k in ("PA2x7_", "washer", "spacer") for i in range(1, 5)]
           + [f"upper__{{S}}roll_horn_{k}{i}" for k in ("M3x6_", "washer") for i in range(1, 5)]
           # the cover posts on stage 2 are dropped (cover redesigned with the shell): their M3 screws / nuts go too
           + [f"{{S}}upper_arm_shell_{k}_{h}" for k in ("front", "rear") for h in ("M3x8", "M3_nut")])


def manifest():
    import copy
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    import interfaces as I
    import screw_plan as SP
    man = json.loads(MANIFEST.read_text())
    cad = json.loads((OUT / "cad-manifest.json").read_text())["parts"]
    by = {p["name"]: p for p in man["parts"]}
    removed = {r.replace("{S}", s) for r in REMOVED for s in ("R_", "L_")}
    parts, log = [], []
    for p in man["parts"]:
        n = p["name"]
        if n in removed:
            log.append(f"{n}: removed (replaced by a screw generated from its defined hole, or no longer used)"); continue
        if n in cad:
            r = cad[n]; q = copy.deepcopy(p)
            for k in ("native_solid_count", "native_validity", "volume_mm3", "mass_kg", "mass_basis"):
                q.pop(k, None)
            q.update(native_step=r["native_step"], native_step_sha256=r["native_step_sha256"], mesh=r["mesh"], mesh_sha256=r["mesh_sha256"],
                     mesh_units="mm", bounds_mm=r["bounds_mm"], volume_mm3=r["volume_mm3"], native_solid_count=r["solids"],
                     mass_kg=round(r["mass_g"] / 1000.0, 6),
                     mass_basis=(f"native STEP volume x {r['density_g_cm3']} g/cm3 ({r['material']}, part setting material)"
                                 if "density_g_cm3" in r else "304 SS sleeves, native STEP volume x 7.93 g/cm3"),
                     shoulder_u="rebuilt by robot/body/arm/shoulder_clevis.py build (owner 2026-11-20, blue / green U)")
            parts.append(q); log.append(f"{n}: rebuilt"); continue
        parts.append(p)
    for s in ("R_", "L_"):
        n = IDLER.replace("upper__R_", "upper__" + s); r = cad[n]
        q = copy.deepcopy(by[f"upper__{s}roll_C047_main_horn"])
        q.update(name=n, native_step=r["native_step"], native_step_sha256=r["native_step_sha256"], mesh=r["mesh"], mesh_sha256=r["mesh_sha256"],
                 mesh_units="mm", bounds_mm=r["bounds_mm"], owning_link=f"{s}shoulder_pitch_link", role="output",
                 shoulder_u="STS3250 STEP member idler (solid 12) placed with the registered roll case transform; mass in the servo's catalogue mass")
        parts.append(q); log.append(f"{n}: added")
    body = F.Body(dict(man, parts=parts))
    need = {f"upper__{s}roll_C047_{m}" for s in ("R_", "L_") for m in ("case", "main_horn", "rear_idler")}
    sub = dict(man, parts=[p for p in parts if p["name"] in need or p.get("motor")])
    iface = I.instances(F.Body(sub), sub)
    inst = {r["part"]: {k: r[k] for k in ("fit_rms_mm", "member", "mirrored", "modelled_with", "hole_snap")} for r in iface["instances"] if r["part"] in need}
    steel = 7.85e-3
    srows, plans = [], []
    for s in ("R_", "L_"):
        for conn, spec in SPECS.items():
            for pl in SP.plan_connector(body, iface, conn, s, spec=spec):
                m = SP.screw_mesh(pl); nm = pl["name"]
                f = PARTS_DIR / f"{nm}.npz"
                np.savez_compressed(f, vertices=np.asarray(m.vertices, float), faces=np.asarray(m.faces, np.int64))
                lbl = f"{pl['thread_label']}x{pl['length_mm']:g}"
                keep = ("hole_id", "thread_label", "head_standard", "length_mm", "stack_mm", "head_gap_mm", "engagement_mm",
                        "reference_engagement_mm", "hole_depth_mm", "tip_margin_mm", "shortfall_mm", "overrun_mm", "stack_change_to_fit_mm")
                srows.append(dict(name=nm, physical_part_id=nm, bom_group=pl["bom_group"], owning_link=pl["owning_link"], role="fastener_proxy",
                                  mesh=_rel(f), mesh_sha256=_sha(f), mesh_units="mm",
                                  bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()],
                                  volume_mm3=round(float(m.volume), 3), mass_kg=round(float(m.volume) * steel / 1000.0, 7),
                                  material=f"{pl['head_standard']} {lbl} into {pl['hole_id'].split(':')[1]} ({pl['terminal_material']})",
                                  color=[195, 204, 214, 255], display_group="mechanism", display_groups=["mechanism"],
                                  overlay_id="actuators_and_hardware", parent_assembly="complete_fdm_ape_body_assembly",
                                  inspection_reference=False, mass_basis="mesh volume x 7.85 g/cm3",
                                  shoulder_u="generated from its defined hole by robot/body/fastening/screw_plan.py",
                                  screw_plan={k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in pl.items() if k in keep}))
                plans.append(srows[-1])
    parts += srows
    log += [f"{r['name']}: {r['screw_plan']['thread_label']}x{r['screw_plan']['length_mm']:g} into {r['screw_plan']['hole_id']}" for r in srows]
    import hashlib
    man["parts"] = parts
    man["study_id"] = man["result_id"] = "cove50-shoulder-u"
    man["shoulder_u"] = dict(idle_task="T-2e49mv (parent T-ce88t9)", owner="2026-11-20: build the blue and green U as drawn, then fix collisions",
                             source_manifest=str(MANIFEST.relative_to(ROOT)), source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                             cad_manifest=_rel(OUT / "cad-manifest.json"), registered_instances=inst, screw_specs=SPECS, changes=log,
                             note="Kinematics (joints, ranges) unchanged. The pitch-horn interface, its screws, the rocker screws and "
                                  "the yaw-case clamp hardware are today's.")
    CANDIDATE.write_text(json.dumps(man, indent=1) + "\n")
    print(f"wrote {CANDIDATE.relative_to(ROOT)} ({len(parts)} parts)")
    print("instances:", json.dumps(inst))
    for r in srows:
        if r["name"].startswith("R_"):
            sp = r["screw_plan"]
            print(f"  {r['name']:36s} {sp['thread_label']}x{sp['length_mm']:g} {sp['hole_id']:44s} stack {sp['stack_mm']} gap {sp['head_gap_mm']} "
                  f"eng {sp['engagement_mm']} (ref {sp['reference_engagement_mm']}) tip margin {sp['tip_margin_mm']} shortfall {sp['shortfall_mm']} overrun {sp['overrun_mm']}")


# ---------------------------------------------------------------------------------------------- standing overlaps
NEW_R = [S1, S1 + "_steel_sleeves", S2, S2 + "_steel_sleeves", IDLER]


def _subjects(man):
    return NEW_R + [p["name"] for p in man["parts"] if p["name"].startswith("R_shoulder_u_")]


def collide(write=True):
    """Exact standing overlap volume (manifold3d booleans) of every new R part (stages, sleeves, idler, new screws)
    against every other part whose box comes within 1 mm; also the overlap meshes (for the collision image)."""
    import manifold3d as M
    import trimesh
    man = json.loads(CANDIDATE.read_text())
    body = F.Body(man)
    subj = _subjects(man)

    def mf(m):
        return M.Manifold(M.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    cache, rows, meshes = {}, [], []
    def get(n):
        if n not in cache:
            m = body.mesh(n)
            cache[n] = mf(m) if m.is_watertight else None
        return cache[n]
    names = list(body.parts)
    for a in subj:
        ma = body.mesh(a); lo, hi = ma.bounds
        for b in names:
            if b == a or (b in subj and subj.index(b) < subj.index(a)):
                continue
            pb = body.parts[b]
            mb = body.mesh(b)
            blo, bhi = mb.bounds
            if np.any(blo > hi + 1.0) or np.any(bhi < lo - 1.0):
                continue
            A, B = get(a), get(b)
            if A is None or B is None:
                rows.append(dict(a=a, b=b, overlap_mm3=None, note="not watertight")); continue
            X = A ^ B
            v = float(X.volume())
            if v > 1e-3:
                xm = X.to_mesh(); tm = trimesh.Trimesh(np.asarray(xm.vert_properties)[:, :3], np.asarray(xm.tri_verts), process=False)
                rows.append(dict(a=a, b=b, link_a=body.parts[a]["owning_link"], link_b=pb["owning_link"], overlap_mm3=round(v, 2),
                                 depth_box_mm=np.round(tm.bounds[1] - tm.bounds[0], 2).tolist(), centre_mm=np.round(tm.bounds.mean(0), 1).tolist()))
                meshes.append((a, b, tm))
    rows.sort(key=lambda r: -(r["overlap_mm3"] or 0))
    if write:
        (OUT / "standing-overlaps.json").write_text(json.dumps(dict(manifest=_rel(CANDIDATE), rows=rows), indent=1) + "\n")
        for r in rows:
            print(f"  {r['a']:44s} x {r['b']:44s} {r['overlap_mm3']} mm3  box {r.get('depth_box_mm')}  at {r.get('centre_mm')}")
    return rows, meshes


def _is_collision(a, b, man):
    """Pairs drawn red: not a generated screw in the standard part it threads into (modelled thread overlap), not a
    part with its own sleeves, not today's unchanged pitch-horn screws / washers (stale hardware models of the kept
    pitch interface, the same overlap as today)."""
    by = {p["name"]: p for p in man["parts"]}
    for s, o in ((a, b), (b, a)):
        hid = (by.get(s, {}).get("screw_plan") or {}).get("hole_id", "")
        if hid and hid.split(":")[0] == o:
            return False
        if o == s + "_steel_sleeves":
            return False
        if o.startswith("upper__R_pitch_horn_") or o.startswith("upper__R_pitch_central_rocker"):
            return False
    return True


# ---------------------------------------------------------------------------------------------- owner images
VIEW_LO, VIEW_HI = np.array([74.0, -26.0, 314.0]), np.array([142.0, 34.0, 380.0])
COL = {"case": (0.30, 0.30, 0.32), "horn": (0.62, 0.45, 0.82), "idler": (0.95, 0.60, 0.15), "blue": (0.15, 0.40, 0.90),
       "green": (0.20, 0.70, 0.25), "sleeve": (0.80, 0.80, 0.83), "screw": (0.55, 0.57, 0.60), "grey": (0.86, 0.86, 0.86),
       "pitchcase": (0.45, 0.45, 0.47)}


def _render(scene, d, up, lo, hi, px_mm=0.12, red=None):
    """Opaque orthographic render by ray casting (embree): first hit per pixel, Lambert + ambient shading, dark outlines
    where the hit part changes or the depth jumps; `red` meshes are drawn over everything (seen through)."""
    import trimesh
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    u = np.cross(d, up); u /= np.linalg.norm(u); v = np.cross(u, d)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pu, pv = C @ u, C @ v
    W, H = int((pu.max() - pu.min()) / px_mm), int((pv.max() - pv.min()) / px_mm)
    gu = pu.min() + (np.arange(W) + 0.5) * px_mm; gv = pv.max() - (np.arange(H) + 0.5) * px_mm
    GU, GV = np.meshgrid(gu, gv)
    c = 0.5 * (lo + hi); R = 0.5 * float(np.abs(d) @ (hi - lo))      # rays start on the near side of the view box
    org = GU.reshape(-1, 1) * u + GV.reshape(-1, 1) * v + (float(c @ d) - R) * d
    dirs = np.tile(d, (len(org), 1))
    ms = [m for m, _ in scene]
    cat = trimesh.util.concatenate(ms); fid = np.concatenate([np.full(len(m.faces), i) for i, m in enumerate(ms)])
    img = np.ones((H * W, 3)); pid = -np.ones(H * W, int); dep = np.full(H * W, np.inf)
    L = -d + 0.6 * v + 0.3 * u; L /= np.linalg.norm(L)
    def cast(mesh):
        rx = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        idx_tri, idx_ray, loc = rx.intersects_id(org, dirs, multiple_hits=False, return_locations=True)
        return idx_tri, idx_ray, loc
    tri, ray, loc = cast(cat)
    n = cat.face_normals[tri]; shade = 0.55 + 0.45 * np.abs(n @ L)
    cols = np.array([scene[i][1] for i in fid[tri]])
    img[ray] = np.clip(cols * shade[:, None], 0, 1); pid[ray] = fid[tri]; dep[ray] = (loc - org[ray]) @ d
    if red is not None and len(red.faces):
        tri, ray, loc = cast(red)
        n = red.face_normals[tri]; shade = 0.65 + 0.35 * np.abs(n @ L)
        img[ray] = np.clip(np.array([1.0, 0.05, 0.05]) * shade[:, None], 0, 1); pid[ray] = 10_000
    img = img.reshape(H, W, 3); pid = pid.reshape(H, W); dep = dep.reshape(H, W)
    edge = np.zeros((H, W), bool)
    for ax in (0, 1):
        dp = np.diff(pid, axis=ax) != 0
        dd = np.abs(np.diff(np.where(np.isfinite(dep), dep, 1e4), axis=ax)) > 1.2
        e = dp | dd
        if ax == 0:
            edge[1:] |= e; edge[:-1] |= e
        else:
            edge[:, 1:] |= e; edge[:, :-1] |= e
    img[edge] = img[edge] * 0.35
    return img, (pu.min(), pu.max(), pv.min(), pv.max())


VIEWS4 = [("TOP (from above): x right, y up (as your sketch)", (0, 0, -1), (0, 1, 0), "x mm (+ = outward)", "y mm (- = face)"),
          ("FRONT (from the face, -y)", (0, 1, 0), (0, 0, 1), "x mm (+ = outward)", "z mm"),
          ("SIDE (from outside, +x)", (-1, 0, 0), (0, 0, 1), "y mm (- = face)", "z mm"),
          ("3/4 view (from front, outside, above)", (-0.75, 0.9, -0.55), (0, 0, 1), None, None)]


def _scene(body, names_cols):
    return [(body.mesh(n), c) for n, c in names_cols]


def views():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import trimesh
    man = json.loads(CANDIDATE.read_text()); body = F.Body(man)
    screws = [p["name"] for p in man["parts"] if p["name"].startswith("R_shoulder_u_")]
    base = [("upper__R_roll_C047_case", COL["case"]), ("upper__R_roll_C047_main_horn", COL["horn"]), (IDLER, COL["idler"]),
            ("upper__R_pitch_C047_main_horn", COL["horn"]), (S1, COL["blue"]), (S1 + "_steel_sleeves", COL["sleeve"]),
            (S2, COL["green"]), (S2 + "_steel_sleeves", COL["sleeve"])] + [(n, COL["screw"]) for n in screws]
    legend = [("roll servo case", COL["case"]), ("roll horn / pitch horn", COL["horn"]), ("roll idler (new)", COL["idler"]),
              ("stage 1 (blue U, pitch link)", COL["blue"]), ("stage 2 (green U, roll link)", COL["green"]),
              ("steel sleeves", COL["sleeve"]), ("new screws", COL["screw"])]
    VIEWS.mkdir(parents=True, exist_ok=True)
    rows, xm = collide(write=False)
    red_x = [m for a, b, m in xm if _is_collision(a, b, man)]
    red = trimesh.util.concatenate(red_x) if red_x else None
    extra = [("center_front_shell", COL["grey"]), ("center_back_shell", COL["grey"]), ("R_upper_arm_hard_cover", COL["grey"]),
             ("upper__R_yaw_C047_case", COL["grey"]), ("upper__R_pitch_C047_case", COL["grey"])]
    for fname, scene_names, title, red_m, leg in (
            ("shoulder-u-parts.png", base, "Right shoulder, new parts as built (standing). Blue = stage 1 U on the pitch link, "
             "green = stage 2 U on the roll link, orange = new roll idler.", None, legend),
            ("shoulder-u-collisions.png", base + extra, "Same parts with the torso shells, upper-arm cover, yaw and pitch "
             "cases in light grey. RED = interpenetration at standing, part vs shell / cover / case / other link\n(drawn on top, seen through the grey; screw threads in their holes left out).", red,
             legend + [("torso shells / arm cover / yaw + pitch case", COL["grey"]), ("interpenetration (standing)", (1.0, 0.05, 0.05))])):
        sc = _scene(body, scene_names)
        fig, axs = plt.subplots(2, 2, figsize=(16, 16), dpi=90)
        for ax, (t, d, up, xl, yl) in zip(axs.ravel(), VIEWS4):
            img, ext = _render(sc, d, up, VIEW_LO, VIEW_HI, red=red_m)
            ax.imshow(img, extent=ext, interpolation="bilinear")
            ax.set_title(t, fontsize=13)
            if xl:
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.tick_params(labelsize=9)
                for s in ax.spines.values():
                    s.set_color("#bbbbbb")
            else:
                ax.axis("off")
            if d == (0, 0, -1):    # roll axis (along y at x 106) and pitch axis (along x at y 9)
                ax.axvline(ROLL_XZ[0], color="#ff7f0e", lw=1.0, ls="--"); ax.axhline(9.0, color="#d62728", lw=1.0, ls="--")
        fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in leg], loc="lower center", ncol=4, fontsize=12, frameon=False)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(VIEWS / fname, facecolor="white")
        plt.close(fig)
        print(f"wrote {(VIEWS / fname).relative_to(ROOT)}")


if __name__ == "__main__":
    {"space": space, "build": build, "manifest": manifest, "collide": collide, "views": views}[sys.argv[1] if len(sys.argv) > 1 else "space"]()
