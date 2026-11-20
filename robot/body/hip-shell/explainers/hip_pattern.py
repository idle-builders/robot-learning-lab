"""Why other humanoids' hips look different from ours, and what that pattern would take here (Idle T-czv2d8, A-16/P-03).

Row 1: our hip as built (real CAD, motors and connectors coloured by link). The leg hangs from the torso through a
       vertical yaw servo and a 51 mm neck: the torso enters the hip RADIALLY from above.
Row 2: the same servos re-arranged in the order G1 / GR-2 / H2 / Booster use, pitch first: the hip-pitch servo is
       mounted on the torso (where it already lies, along the pitch axis), the roll stage stays as it is, the yaw
       servo moves down into the thigh with its shaft on the thigh axis, and the knee servo turns 90 deg about the
       knee axis to make room. The torso then enters the hip AXIALLY, along the pitch axis. Block sketch, true scale.
Row 3: what that means for a hip globe in the side view: radial entry crosses the globe and pitch sweeps it (nested
       bands needed, A-08); axial entry does not cross the globe in the side view at all (one piece, one round seam).

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/explainers/hip_pattern.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "sections"))
import kit  # noqa: E402
from joint_cuts import skeleton_parts  # noqa: E402
from leg_motion import LINK_COL, ortho, solid  # noqa: E402

C = np.array([59.0, 2.0, 219.0])
ROLL = np.array([56.0, 2.0, 190.0])
KNEE = np.array([52.0, 4.0, 123.0])
SIDE_LIM = ((-110, 110), (85, 330)); FRONT_LIM = ((-150, 40), (85, 330))


def axis_stats(body):
    """Each hip servo body and its child bracket, measured about that joint's own axis (full CAD meshes)."""
    J = {j["name"]: j for j in body.child_joint.values()}

    def about(name, jn):
        o = np.asarray(J[jn]["origin_world_mm"], float); a = np.asarray(J[jn]["axis_world_neutral"], float)
        a /= np.linalg.norm(a)
        m = body.mesh(name); d = np.asarray(m.vertices) - o; t = d @ a
        r = np.linalg.norm(d - np.outer(t, a), axis=1)
        c = np.asarray(m.centroid) - o
        return dict(r_max=float(r.max()), axial=[float(t.min()), float(t.max())],
                    centroid_offset=float(np.linalg.norm(c - (c @ a) * a)))
    rows = {"R_hip_yaw": ("lower__R_yaw_servo_case_reference", "R_hip_yaw_to_pitch_combined"),
            "R_hip_pitch": ("R_hip_pitch_ST3025_case", "R_hip_pitch_to_roll_combined"),
            "R_hip_roll": ("R_roll_servo_case_reference", "R_thigh_combined")}
    return {jn: dict(servo=s, servo_about_own_axis=about(s, jn), child=c, child_about_axis=about(c, jn))
            for jn, (s, c) in rows.items()}


def box(lo, hi):
    return trimesh.creation.box(bounds=[lo, hi])


def pattern_layout():
    """Pitch-first block sketch, right leg (mm, standing). Same servos, same pitch and roll positions as today."""
    return [
        # torso
        ("hip-pitch servo ST3025, now on the torso", box([14, -8, 209], [57, 12, 249]), "torso"),
        ("battery cells (unchanged)", box([-11, -29, 209], [11, 41, 253]), "torso"),
        # pitch link (moves with pitch only)
        ("pitch horn + bracket (as today)", box([57, -14, 177], [71, 25, 229]), "R_pitch_link"),
        ("roll servo STS3250 (as today)", box([21, -10.5, 178], [66, 25.8, 202]), "R_pitch_link"),
        # roll link = upper thigh
        ("roll bracket / upper thigh", box([18, -16, 170], [70, 30, 178]), "R_roll_link"),
        # lower thigh (new link that turns with yaw): the yaw servo body rides on it, its horn bolts into the roll bracket
        ("hip-yaw servo, moved into the thigh, shaft on the thigh axis", box([21, -10, 133.5], [66, 14, 170.5]), "lower_thigh"),
        ("knee servo ST3025, turned 90 deg about the knee axis", box([32, -6, 113], [76, 34, 133]), "lower_thigh"),
    ]


def draw_axes(ax, view, which):
    """Axis lines and turning points in an orthographic view (side: u = -y, v = z; front: u = -x, v = z)."""
    def uv(p):
        return (-p[1], p[2]) if view == "side" else (-p[0], p[2])
    if "yaw_top" in which:
        a, b = uv([59, 2, 330]), uv([59, 2, 160])
        ax.plot([a[0], b[0]], [a[1], b[1]], "-.", color="#1a9850", lw=2)
        ax.text(a[0] + 3, 318, "yaw axis", color="#1a9850", fontsize=11)
    if "yaw_thigh" in which:
        a, b = uv([56, 2, 185]), uv([56, 2, 100])
        ax.plot([a[0], b[0]], [a[1], b[1]], "-.", color="#1a9850", lw=2)
        ax.text(a[0] + 3, 104, "yaw axis (in the thigh)", color="#1a9850", fontsize=11)
    c = uv(C); r = uv(ROLL)
    ax.plot(*c, "+", color="#d00", ms=18, mew=3)
    ax.plot(*r, "x", color="#2166ac", ms=12, mew=3)
    if view == "side":
        ax.plot([-110, 110], [219, 219], ":", color="#d00", lw=0.8)
    else:
        ax.plot([-2, -2], [100, 330], ":", color="#2166ac", lw=0.8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Wedge
    body = kit.base()
    parts = skeleton_parts(body)
    sol = {n: solid(body, n) for n in parts}
    stats = axis_stats(body)
    (HERE / "hip-pattern-axis-stats.json").write_text(json.dumps(stats, indent=1))
    fig, axs = plt.subplots(3, 3, figsize=(33, 31))
    q = dict(body.standing)
    items = []
    for n, p in parts.items():
        lk = p["owning_link"]
        if lk.startswith("L_"):
            continue
        items.append((sol[n], body.link_T(lk, q), LINK_COL.get(lk, np.array([0.7, 0.7, 0.7])), 1.0))
    # ---------------- row 1: ours
    for k, (view, lim) in enumerate((("side", SIDE_LIM), ("front", FRONT_LIM))):
        ax = axs[0, k]
        ortho(ax, items, view, lim); ax.axis("on"); ax.grid(alpha=0.2)
        draw_axes(ax, view, {"yaw_top"})
        if view == "side":
            ax.add_patch(Circle((-2, 219), 47.8, fill=False, ec="#e6550d", lw=2.5, ls="--"))
            ax.annotate("pitch-to-roll bracket reaches r 48\nfrom the pitch axis and swings with it", (40, 190), (55, 140),
                        fontsize=12, color="#e6550d", arrowprops=dict(arrowstyle="->", color="#e6550d"))
            ax.annotate("51 mm neck: the torso reaches the hip\nfrom ABOVE (radial entry)", (-2, 250), (-105, 305),
                        fontsize=12, color="#1a9850", arrowprops=dict(arrowstyle="->", color="#1a9850", lw=2))
            ax.set_title("A  Our hip as built, side view (yaw -> pitch -> roll)", fontsize=16)
        else:
            ax.annotate("hip-pitch servo lies 45 mm inboard\nalong the pitch axis, but rides on the\nyaw link: "
                        "yaw swings it r 46 under\nthe pelvis (cradle contacts)", (-36, 230), (-148, 300), fontsize=12,
                        color="#333", arrowprops=dict(arrowstyle="->"))
            ax.annotate("roll axis 29 mm below C", (-56, 190), (-148, 160), fontsize=12, color="#2166ac",
                        arrowprops=dict(arrowstyle="->", color="#2166ac"))
            ax.add_patch(Circle((-56, 190), 66.7, fill=False, ec="#2166ac", lw=2, ls="--"))
            ax.set_title("B  Our hip as built, front view", fontsize=16)
    ax = axs[0, 2]; ax.axis("off")
    rows = [("joint", "servo body about its own axis", "child bracket about the axis"),
            ("hip yaw", f"r {stats['R_hip_yaw']['servo_about_own_axis']['r_max']:.0f}, off-centre "
                        f"{stats['R_hip_yaw']['servo_about_own_axis']['centroid_offset']:.0f}",
             f"neck r {stats['R_hip_yaw']['child_about_axis']['r_max']:.0f}, 51 mm long"),
            ("hip pitch", f"r {stats['R_hip_pitch']['servo_about_own_axis']['r_max']:.0f}, off-centre "
                          f"{stats['R_hip_pitch']['servo_about_own_axis']['centroid_offset']:.0f}",
             f"r {stats['R_hip_pitch']['child_about_axis']['r_max']:.0f} (wraps the roll servo)"),
            ("hip roll", f"r {stats['R_hip_roll']['servo_about_own_axis']['r_max']:.0f}, off-centre "
                         f"{stats['R_hip_roll']['servo_about_own_axis']['centroid_offset']:.0f}",
             f"r {stats['R_hip_roll']['child_about_axis']['r_max']:.0f} (thigh to the knee)"),
            ("pancake actuator", "r = D/2, off-centre 0", "flange at r <= D/2")]
    txt = "\n".join(f"{a:18s} {b:28s} {c}" for a, b, c in rows)
    ax.text(0.0, 1.0, "C  Where the difference comes from (measured on our CAD)\n\n" + txt +
            "\n\n1. Attachment. G1, GR-2, H2, Booster mount the FIRST hip actuator on the pelvis along its\n"
            "   own axis (a pitch disc at the pelvis side; Walker S2 and Digit: a roll disc). The leg\n"
            "   meets the torso through one rotation about that axis, so the seam is a circle that\n"
            "   never moves. Ours meets the torso through a vertical yaw servo and a 51 mm neck, so the\n"
            "   pitch servo, the pitch bracket and the whole leg swing under the pelvis.\n"
            "2. Actuator shape. Pancake actuators are discs centred on their axis. Our bus servos are\n"
            "   bricks with the shaft 10 mm from one end: off-centre 9-14 mm, reaching r 31-37 about\n"
            "   their own axis, so a coaxial housing around one is a drum of D 62-74.\n"
            "3. Brackets. Each child link straddles its parent servo with a U-bracket (horn + idler),\n"
            "   which reaches r 48 (pitch) and r 67 (roll) and sweeps the side and front views.\n"
            "4. Axis order. Yaw-first puts a radial neck into the side view (A-08: 4-5 bands front,\n"
            "   3 back). Pitch-first enters along the pitch axis and leaves the side view free.\n\n"
            "Pancakes are not required for the look: at our torque class the round actuators in\n"
            "robot/actuators/parts.csv are drums too (AK40-10 D53x40, AK45-36 D55x57, GIM4310-36\n"
            "D55x47; 185-350 g; USD 136-186 + CAN), about the size of a brick in a coaxial drum.\n"
            "The attachment and the order are what make the pattern.",
            fontsize=13.5, family="monospace", va="top", transform=ax.transAxes)
    # ---------------- row 2: pattern with our bricks
    lay = pattern_layout()
    col = dict(LINK_COL, lower_thigh=np.array([0.2, 0.65, 0.45]))
    for k, (view, lim) in enumerate((("side", SIDE_LIM), ("front", FRONT_LIM))):
        ax = axs[1, k]
        pitems = [(m, np.eye(4), col.get(l, np.array([0.7, 0.7, 0.7])), 1.0) for nm, m, l in lay
                  if view == "front" or not nm.startswith("battery")]   # the cells sit at x +-11, far inboard
        ortho(ax, pitems, view, lim); ax.axis("on"); ax.grid(alpha=0.2)
        draw_axes(ax, view, {"yaw_thigh"})
        if view == "side":
            ax.add_patch(Circle((-2, 219), 48, fill=False, ec="#e6550d", lw=3))
            ax.text(-105, 300, "hip globe r 48 on the pitch link: the torso holds the\npitch servo along the axis, "
                    "so nothing static\ncrosses the globe in this view", fontsize=12, color="#e6550d")
            ax.set_title("D  Pattern with our servos (pitch -> roll -> yaw), side view, block sketch\n"
                         "grey = torso, orange = pitch link, blue = roll bracket, green = lower thigh (turns with yaw)",
                         fontsize=15)
        else:
            ax.annotate("pitch servo on the torso, along the axis:\nthe torso enters the hip AXIALLY", (-36, 229),
                        (-148, 305), fontsize=12, color="#555", arrowprops=dict(arrowstyle="->", lw=2))
            ax.annotate("yaw servo in the thigh, shaft on the thigh axis\n(45 x 24 x 37, r 37 about it: fits the\n"
                        "today's 77 x 63 thigh section)", (-44, 152), (-148, 110), fontsize=12, color="#1a9850",
                        arrowprops=dict(arrowstyle="->", color="#1a9850"))
            ax.annotate("knee servo turned 90 deg about the knee axis\n(body backwards instead of up): frees 20 mm",
                        (-54, 123), (-30, 95), fontsize=11, color="#6a3d9a", arrowprops=dict(arrowstyle="->", color="#6a3d9a"))
            ax.set_title("E  Pattern with our servos, front view", fontsize=16)
    ax = axs[1, 2]; ax.axis("off")
    ax.text(0.0, 1.0, "F  What the pattern would take here (P-03), and its limits\n\n"
            "Kept: all servo types and tiers (brick family, one Feetech bus), the\n"
            "  hip-pitch servo position, the whole pitch -> roll stage, hip width,\n"
            "  hip centre C, thigh and shin lengths.\n"
            "Changed: hip yaw moves from the torso (above the pitch servo) into the\n"
            "  thigh (below the roll servo); the pitch servo mounts on the battery\n"
            "  cradle; the 51 mm yaw neck and the yaw servo in the torso go; the knee\n"
            "  servo turns 90 deg; new thigh connectors (roll -> yaw, yaw -> knee).\n\n"
            "Thigh length budget, C to knee axis 96 mm:\n"
            "  roll stage down to z 170; yaw servo body 37 mm below it (its horn\n"
            "  bolts into the roll bracket) -> z 133.5; knee servo turned 90 deg:\n"
            "  top at z 133  ->  about 0-5 mm to spare, depending on the horn.\n"
            "  With the knee servo upright as built (top z 153): 18 mm short.\n\n"
            "Limits and costs:\n"
            "  - kinematics change: the yaw axis follows the thigh (tilts with\n"
            "    pitch) -> new URDF and a WBC retrain on the GPU;\n"
            "  - the yaw servo's 55-74 g moves from the torso into the leg;\n"
            "  - roll stays 29 mm below pitch (as on G1, 30 mm), so a single globe\n"
            "    still covers only pitch; roll gets its own round seam lower down;\n"
            "  - the thigh is packed tight (0.5 mm in the block sketch);\n"
            "  - removes the cradle contacts seen with toes-in deep flexion, since\n"
            "    yaw no longer swings the hip bracket under the pelvis (to verify).\n\n"
            "A fully spherical hip (P-01, all three axes through C) with bricks needs\n"
            "a globe of r 55-63 that reaches past the body's centreline at x 59.",
            fontsize=13.5, family="monospace", va="top", transform=ax.transAxes)
    # ---------------- row 3: globe in the side view, radial vs axial entry
    for k, (lab, neck) in enumerate((("G  Radial entry (ours): the torso neck crosses the globe at the top; pitch\n"
                                      "-85..+40 stretches the front and back skin -> 4-5 + 3 nested bands", True),
                                     ("H  Axial entry (pattern): the torso holds the joint along its axis; in the side\n"
                                      "view the globe is one piece that turns in place with the thigh", False))):
        ax = axs[2, k]
        ax.set_xlim(-110, 110); ax.set_ylim(85, 330); ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.add_patch(Circle((0, 219), 48, fc="#fdd0a2", ec="#e6550d", lw=3))
        if neck:
            ax.add_patch(Wedge((0, 219), 54, 90 - 23, 90 + 23, width=14, fc="#7f7f7f"))
            ax.add_patch(Wedge((0, 219), 54, 90 - 23 - 132, 90 - 23, width=6, fc="#2ca02c"))
            ax.add_patch(Wedge((0, 219), 54, 90 + 23, 90 + 23 + 120, width=6, fc="#ff7f0e"))
            ax.text(0, 286, "torso neck 46 deg", ha="center", fontsize=12)
            ax.text(55, 245, "front skin\n48..172 deg", color="#2ca02c", fontsize=12)
            ax.text(-105, 245, "back skin\n80..206 deg", color="#ff7f0e", fontsize=12)
        else:
            ax.add_patch(Circle((0, 219), 16, fc="#7f7f7f", ec="k"))
            ax.text(0, 192, "torso holds the\naxis here (inboard)", ha="center", fontsize=11)
            ax.text(55, 260, "no static part crosses\nthe globe in this view", color="#e6550d", fontsize=12)
        ax.add_patch(Wedge((0, 219), 54, -90 - 30, -90 + 30, width=14, fc="#3355aa", alpha=0.85))
        ax.text(0, 150, "thigh flows out of the globe" if not neck else "thigh neck (turns with pitch)", ha="center",
                fontsize=12, color="w" if False else "#3355aa")
        ax.set_title(lab, fontsize=15)
    ax = axs[2, 2]; ax.axis("off")
    ax.text(0.0, 1.0, "Reading the reference images with this\n\n"
            "GR-2: pitch first, tilted 30 deg; the round disc in each hip is the\n"
            "  pitch housing, held by the V-shaped pelvis along its axis.\n"
            "G1: pitch disc at the pelvis side (axial); the white thigh shell\n"
            "  starts as a dome around it; roll 30 mm lower, yaw in the thigh.\n"
            "Walker S2: roll first; the black hip blocks hang from the ends of\n"
            "  the pelvis bar along the roll axis.\n"
            "Digit: roll and yaw housings at the pelvis sides, axial pods.\n\n"
            "In every case the first hip actuator is held by the pelvis along\n"
            "its own axis, and each next stage is a housing round about its own\n"
            "axis. That is the common pattern; ours starts with a vertical\n"
            "servo above the hip and a neck down to it.",
            fontsize=13.5, family="monospace", va="top", transform=ax.transAxes)
    fig.suptitle("A-16 / P-03: why the hips of G1, GR-2, Walker S2 and Digit look different from ours, and what that "
                 "pattern would take with our servos (right leg, standing, mm; red + = hip centre C, blue x = roll axis)",
                 fontsize=19)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / "hip-pattern.png"
    plt.savefig(out, dpi=50); print(out)


if __name__ == "__main__":
    main()
