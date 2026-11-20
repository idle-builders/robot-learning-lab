"""Package and publish the shoulder roll U connectors (Idle T-2e49mv) on the robot design dashboard.

The shoulder roll servo (STS3250) is held on both faces: stage 1 (blue U, pitch link) grips the roll main horn and a new roll
rear idler, stage 2 (green U, roll link) clamps the roll case on its 2 + 2 far holes and keeps today's yaw-case clamp.
Owner 2026-11-20 (M-rya3ty): "Much better the two U connectors look clean now. Continue". A new architecture, so its own
study (not a version of cove50-arm-fdm-combined). Inputs: the candidate manifest written by shoulder_clevis.py
(build + manifest + views) and the shared check reports (robot/body/checks/run.py) named in REPORTS.

  python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 900 -- .venv/bin/python -B robot/body/arm/publish_shoulder_u.py package
  .venv/bin/python -B robot/body/arm/publish_shoulder_u.py record     # writes the publication JSON from the check reports
  .venv/bin/python infra/dashboard/results_hub.py publish artifacts/robot/body/arm/shoulder-clevis/publication.json

package: assembly GLB of the candidate body, bilateral STEP of the new printed parts with their sleeves, copies of the owner
images and the check reports into the study folder (the dashboard serves only registered files under artifacts/), the pitch
output moments over the arm load envelope, and the FEM distribution chart. Everything here is derived and recreated by this
script. Every number is a measurement in native units; nothing accepts or rejects.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ART = ROOT / "artifacts/robot/body/arm/shoulder-clevis"
MANIFEST = ART / "assembly-parts-shoulder-u.json"
CHECKS = ROOT / ".cache/work/cad-speed/checks"
VIEWS = ROOT / ".cache/work/shoulder-views"
STUDY = "shoulder-roll-u-connectors"
S1, S2, IDLER = "shoulder_pitch_to_roll_combined", "shoulder_roll_to_yaw_combined", "roll_C047_rear_idler"
NAMES = {S1: "Blue U (stage 1, pitch link)", S2: "Green U (stage 2, roll link)"}
MATS = ("Bambu PAHT-CF", "Bambu PETG-CF")            # datasheet materials the setups ran (the legacy_* entry only reproduces old numbers)
CASES = {"recovery_4kg_1p5g": "recovery 4 kg x 1.5 g (58.8 N)", "support_4kg_1g": "support 4 kg x 1 g (39.2 N)",
         "manip_1p5kg_reach": "manipulation 1.5 kg x 1 g (14.7 N)"}
# label -> what it is; the reports come from run.py on the candidate manifest (before = the selected body at the time of the run)
REPORTS = {
    "shoulder-u-verify3": "R: geometry, mass, gaps, static_audit, clearance (library), press, screws, holes, walls, overhang, bed, driver_access, range_sweep",
    "shoulder-u-verify-L": "L: the same checks",
    "shoulder-u-s1-fem-final": "R blue U FEM, arm load envelope, h 0.9 (variant shoulder_u)",
    "shoulder-u-s1-fem-h07b": "R blue U FEM, recovery / PAHT-CF at h 0.7 (convergence)",
    "shoulder-u-s2-fem-i3": "R green U FEM, arm load envelope, h 1.2 (variant shoulder_u)",
    "shoulder-u-s2-fem-h09": "R green U FEM, arm load envelope, h 0.9 (convergence)",
    "shoulder-u-fem-L": "L blue + green U FEM, arm load envelope (blue h 0.9, green h 1.2)",
    "arm-envelope-s1": "before: today's stage 1 in the selected body, same envelope, h 1.2",
    "arm-envelope-s2": "before: today's stage 2 in the selected body, same envelope, h 1.2",
}
RENDERS = ("shoulder-u-parts.png", "shoulder-u-collisions.png", "shoulder-today.png")
CHART = ART / "renders/fem-envelope-before-after.png"
MOMENTS = ART / "pitch-output-moments.json"
SHELLS = ("center_front_shell", "center_back_shell", "R_upper_arm_hard_cover", "L_upper_arm_hard_cover")


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def report(label):
    return json.loads((ART / "checks" / f"{label}.json").read_text())


# ------------------------------------------------------------------------------------------------------------- package
def package():
    (ART / "checks").mkdir(exist_ok=True)
    for label in REPORTS:
        shutil.copy2(CHECKS / label / "report.json", ART / "checks" / f"{label}.json")
    (ART / "renders").mkdir(exist_ok=True)
    for n in RENDERS:
        shutil.copy2(VIEWS / n, ART / "renders" / n)
    pitch_moments()
    chart()
    sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
    from package import scene  # motor-refit packager: one named node per part, metres
    man = json.loads(MANIFEST.read_text())
    scene(man).export(str(ART / "assembly-shoulder-u.glb"))
    from build123d import Compound, export_step, import_step
    shapes = [import_step(str(ART / "parts" / f"{s}{n}{suffix}.step"))
              for s in ("R_", "L_") for n in (S1, S2) for suffix in ("", "_steel_sleeves")]
    export_step(Compound(children=shapes), str(ART / "connectors-bilateral.step"))
    print("packaged", rel(ART))


def pitch_moments():
    """The pitch servo stays held on its output side only: the moment the arm load envelope puts on the pitch joint, split into
    the part about the pitch axis (capped at stall by the envelope) and the off-axis part the servo's output bearing carries."""
    sys.path.insert(0, str(ROOT / "robot/body"))
    from checks import arm_loads as A
    from checks import bodylib as F
    man = json.loads(MANIFEST.read_text())
    body = F.Body(man)
    setup = json.loads((ROOT / "robot/body/checks/data/part_settings.json").read_text())["R_" + S1]["fem"]["load_cases"]
    j = next(j for j in man["joints"] if j["name"] == "R_shoulder_pitch")
    o = np.asarray(j["origin_world_mm"], float); a = np.asarray(j["axis_world_neutral"], float); a /= np.linalg.norm(a)
    out = dict(what="moment of the arm load envelope (hand force, +-30 deg cone, stall-capped) at the R_shoulder_pitch joint origin, "
                    "in the pitch link frame: about the pitch axis (servo torque) and off-axis (carried by the servo output only)",
               joint_origin_mm=o.tolist(), axis=a.tolist(), manifest=rel(MANIFEST), cases={})
    q = lambda x: {k: round(float(np.quantile(x, v)) / 1000.0, 2) for k, v in (("min", 0), ("p10", .1), ("p50", .5), ("p90", .9), ("max", 1))}
    for case in CASES:
        W, _, S = A.envelope(body, setup[case]["hand_envelope"], o)
        M = W[:, 3:]; ax = M @ a
        out["cases"][case] = dict(wrenches=len(W), axial_Nm=q(np.abs(ax)), off_axis_Nm=q(np.linalg.norm(M - np.outer(ax, a), axis=1)),
                                  total_Nm=q(np.linalg.norm(M, axis=1)), stall_capped_wrenches=S["stall_capped_wrenches"])
    MOMENTS.write_text(json.dumps(out, indent=1) + "\n")


def layer(p):
    """The layer normal the part ran along: its print axis, else the one stored axis (the before parts print along -y / +y)."""
    return p["layer_direction"].get("print_axis") or next(iter(p["by_layer_normal"]))


def fem_rows():
    """{(part, material, case): {"after": envelope, "before": envelope}} for the R side, each part along its own print direction."""
    rows = {}
    for side, labels in (("after", {S1: "shoulder-u-s1-fem-final", S2: "shoulder-u-s2-fem-i3"}),
                         ("before", {S1: "arm-envelope-s1", S2: "arm-envelope-s2"})):
        for part, label in labels.items():
            p = report(label)["checks"]["fem"]["per_part"]["R_" + part]
            ln = layer(p)
            for m in MATS:
                for c in CASES:
                    rows.setdefault((part, m, c), {})[side] = dict(p["by_layer_normal"][ln][m][c]["envelope"], layer_normal=ln,
                                                                   mesh_h_mm=p["mesh_h_mm"], nodes=p["nodes"])
    return rows


def chart():
    """Distribution over the 8310 envelope wrenches per part, material and case: p99 strength ratio and peak displacement,
    before (today's stage in the selected body) and after. Thin line min..max, bar p10..p90, tick p50."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a65", "#e4e3df"
    rows = fem_rows()
    keys = [(p, m, c) for p in (S1, S2) for m in MATS for c in CASES]
    fig, axs = plt.subplots(1, 2, figsize=(14, 8.2), sharey=True, facecolor="#fcfcfb")
    for ax, metric, xlab, log in ((axs[0], "p99_strength_ratio", "p99 in-plane strength ratio (x, log scale; right = more margin)", True),
                                  (axs[1], "max_displacement_mm", "peak displacement (mm, log scale)", True)):
        ax.set_facecolor("#fcfcfb")
        for i, k in enumerate(keys):
            y = len(keys) - 1 - i
            for side, col, dy in (("before", orange, 0.17), ("after", blue, -0.17)):
                d = rows[k][side][metric]
                ax.plot([d["min"], d["max"]], [y + dy] * 2, color=col, lw=1.2, solid_capstyle="round")
                ax.plot([d["p10"], d["p90"]], [y + dy] * 2, color=col, lw=6, solid_capstyle="round")
                ax.plot([d["p50"]] * 2, [y + dy - 0.13, y + dy + 0.13], color="#fcfcfb", lw=2.2)
        if log:
            ax.set_xscale("log")
        if metric == "p99_strength_ratio":
            ax.axvline(1.0, color=muted, lw=1, ls="--")
            ax.text(1.04, len(keys) - 0.45, "ratio 1", color=muted, fontsize=9, va="bottom")
        ax.set_xlabel(xlab, color=ink, fontsize=10)
        ax.grid(axis="x", color=grid, lw=0.8, which="both"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bdbcb6")
        ax.tick_params(colors=muted, labelsize=9)
    axs[0].set_yticks(range(len(keys)))
    axs[0].set_yticklabels([f"{NAMES[p].split(' (')[0]} · {m.replace('Bambu ', '')} · {CASES[c].split(' (')[0]}" for p, m, c in reversed(keys)],
                           fontsize=9, color=ink)
    fig.legend(handles=[Line2D([], [], color=blue, lw=6, label="after: U connectors (this study)"),
                        Line2D([], [], color=orange, lw=6, label="before: today's stage in the selected body")],
               loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Right shoulder connectors over the arm load envelope (8310 hand-force wrenches per case): thin line min-max, "
                 "bar p10-p90, gap p50", fontsize=11, color=ink, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------------- record
def m(name, value, unit, condition=None):
    out = {"name": name, "value": value, "unit": unit}
    if condition:
        out["condition"] = condition
    return out


def qtext(d, nd=2):
    return f"min {d['min']:.{nd}f}, p10 {d['p10']:.{nd}f}, p50 {d['p50']:.{nd}f}, p90 {d['p90']:.{nd}f}, max {d['max']:.{nd}f}"


def fem_evidence(refs):
    rows = fem_rows()
    share = [100.0 * r["before"]["wrenches_with_p99_strength_ratio_below_1"] / r["before"]["wrenches"]
             for (pt, mt, c), r in rows.items() if c != "manip_1p5kg_reach"]
    below = f"{min(share):.0f}-{max(share):.0f} %"
    rec = []
    for part in (S1, S2):
        for mat in MATS:
            r = rows[(part, mat, "recovery_4kg_1p5g")]; a, b = r["after"], r["before"]
            lab = f"{NAMES[part].split(' (')[0]}, {mat.replace('Bambu ', '')}"
            rec += [m(f"{lab}: p99 strength ratio, worst wrench", a["p99_strength_ratio"]["min"], "x",
                      f"p50 {a['p99_strength_ratio']['p50']:.2f}; before {b['p99_strength_ratio']['min']:.2f} / p50 {b['p99_strength_ratio']['p50']:.2f}"),
                    m(f"{lab}: p99 layer ratio, worst wrench", a["p99_layer_ratio"]["min"], "x",
                      f"p50 {a['p99_layer_ratio']['p50']:.2f}; before {b['p99_layer_ratio']['min']:.2f} / p50 {b['p99_layer_ratio']['p50']:.2f}"),
                    m(f"{lab}: peak displacement, worst wrench", a["max_displacement_mm"]["max"], "mm",
                      f"p50 {a['max_displacement_mm']['p50']:.3f}; before {b['max_displacement_mm']['max']:.1f} / p50 {b['max_displacement_mm']['p50']:.2f}"),
                    m(f"{lab}: p99 von Mises, worst wrench", a["p99_vm_MPa"]["max"], "MPa",
                      f"p50 {a['p99_vm_MPa']['p50']:.1f}; before {b['p99_vm_MPa']['max']:.0f} / p50 {b['p99_vm_MPa']['p50']:.0f}"),
                    m(f"{lab}: wrenches with strength / layer ratio below 1", a["wrenches_with_p99_strength_ratio_below_1"] + a["wrenches_with_p99_layer_ratio_below_1"],
                      f"of {a['wrenches']} (strength + layer)",
                      f"strength {a['wrenches_with_p99_strength_ratio_below_1']}, layer {a['wrenches_with_p99_layer_ratio_below_1']}; before strength "
                      f"{b['wrenches_with_p99_strength_ratio_below_1']}, layer {b['wrenches_with_p99_layer_ratio_below_1']}")]
    other = []
    for part in (S1, S2):
        for mat in MATS:
            for c in ("support_4kg_1g", "manip_1p5kg_reach"):
                a, b = rows[(part, mat, c)]["after"], rows[(part, mat, c)]["before"]
                other.append(m(f"{NAMES[part].split(' (')[0]}, {mat.replace('Bambu ', '')}, {CASES[c].split(' (')[0]}: p99 strength ratio, worst wrench",
                               a["p99_strength_ratio"]["min"], "x",
                               f"p50 {a['p99_strength_ratio']['p50']:.2f}; layer min {a['p99_layer_ratio']['min']:.2f}; displacement max "
                               f"{a['max_displacement_mm']['max']:.3f} mm; below 1: {a['wrenches_with_p99_strength_ratio_below_1']} / "
                               f"{a['wrenches_with_p99_layer_ratio_below_1']}; before min {b['p99_strength_ratio']['min']:.2f}, p50 "
                               f"{b['p99_strength_ratio']['p50']:.2f}, displacement max {b['max_displacement_mm']['max']:.1f} mm, below 1: "
                               f"{b['wrenches_with_p99_strength_ratio_below_1']} / {b['wrenches_with_p99_layer_ratio_below_1']}"))
    # mesh convergence (recovery, PAHT-CF) and the mirrored left side
    def env(label, part, mat="Bambu PAHT-CF", case="recovery_4kg_1p5g"):
        p = report(label)["checks"]["fem"]["per_part"][part]
        return p["by_layer_normal"][layer(p)][mat][case]["envelope"], p["mesh_h_mm"], p["nodes"]
    conv = []
    for part, coarse, fine in ((S1, "shoulder-u-s1-fem-final", "shoulder-u-s1-fem-h07b"), (S2, "shoulder-u-s2-fem-i3", "shoulder-u-s2-fem-h09")):
        (c, hc, nc), (f, hf, nf) = env(coarse, "R_" + part), env(fine, "R_" + part)
        d = lambda k, q: round(100.0 * (f[k][q] - c[k][q]) / c[k][q], 1)
        conv.append(m(f"{NAMES[part].split(' (')[0]}: change of the worst p99 von Mises, h {hc} -> {hf} mm", d("p99_vm_MPa", "max"), "%",
                      f"{nc} -> {nf} nodes; p50 {d('p99_vm_MPa', 'p50'):+} %, peak displacement {d('max_displacement_mm', 'max'):+} %; "
                      f"worst strength ratio {c['p99_strength_ratio']['min']:.2f} -> {f['p99_strength_ratio']['min']:.2f} (recovery, PAHT-CF)"))
    for part in (S1, S2):
        for mat in MATS:
            (r, _, _), (l, hl, _) = env("shoulder-u-s1-fem-final" if part == S1 else "shoulder-u-s2-fem-i3", "R_" + part, mat), env("shoulder-u-fem-L", "L_" + part, mat)
            conv.append(m(f"Left side, {NAMES[part].split(' (')[0]}, {mat.replace('Bambu ', '')}: p99 strength ratio, worst wrench", l["p99_strength_ratio"]["min"], "x",
                          f"recovery, h {hl}, {l['wrenches']} wrenches; p50 {l['p99_strength_ratio']['p50']:.2f}; right {r['p99_strength_ratio']['min']:.2f} / "
                          f"p50 {r['p99_strength_ratio']['p50']:.2f}; below 1: {l['wrenches_with_p99_strength_ratio_below_1']}"))
    s1 = report("shoulder-u-s1-fem-final")["checks"]["fem"]["per_part"]["R_" + S1]
    s2 = report("shoulder-u-s2-fem-i3")["checks"]["fem"]["per_part"]["R_" + S2]
    pm = json.loads(MOMENTS.read_text())["cases"]
    worst1 = s1["by_layer_normal"]["x"]["Bambu PAHT-CF"]["recovery_4kg_1p5g"]["envelope"]["worst"][0]
    worst2 = s2["by_layer_normal"]["z"]["Bambu PAHT-CF"]["recovery_4kg_1p5g"]["envelope"]["worst"][0]
    cond = {"load_cases": "hand-tip force of the case magnitude within a +-30 deg push cone around the hand-to-shoulder line, over every arm pose "
                          "(15 deg grid over the manifest ranges, 648 poses x 96 half-sphere directions -> 8310 wrenches per case), capped where a "
                          "servo of the chain would pass its stall (STS3250 4.9 N m), moved to the load centroid in the stage frame "
                          "(robot/body/checks/arm_loads.py; owner decision M-s8gt2p, Idle T-ysng9c)",
            "setup": "variant shoulder_u in robot/body/checks/data/part_settings.json. Blue U: supports = the 4 sleeve sites at the pitch main horn "
                     "(unchanged interface); load = the sleeve sites at the roll main horn and the new roll idler as one rigid patch. Green U: "
                     "supports at the yaw-case clamp, load at the 2 + 2 roll-case sleeve sites. 304 steel collars 0.7 mm around the sleeve bores.",
            "solver": "gmsh P1 tetrahedra, linear static, unit-wrench basis; p99 = volume-weighted 99th percentile of the element field",
            "print_direction": f"blue {s1['layer_direction']['up']} (L -x), green plate-down (up {s2['layer_direction']['up']}, L mirrored); "
                               "the layer normal of the strength / layer ratios is the print direction",
            "mesh": f"blue h {s1['mesh_h_mm']} mm ({s1['nodes']} nodes), green h {s2['mesh_h_mm']} mm ({s2['nodes']} nodes); before h 1.2",
            "worst_recovery_wrench": f"blue: pose {worst1['pose_deg']}, {worst1['force_N']} N, capped by {worst1['capped_by']}; green: pose "
                                     f"{worst2['pose_deg']}, {worst2['force_N']} N, capped by {worst2['capped_by']}",
            "before": "today's stage 1 / stage 2 in the selected body (roll-flip manifest at the time of the run), same envelope and materials, "
                      "each along its own print direction (y)"}
    lim = ("Linear static FEM of the printed part alone: servo cases, horn and idler rigid at the patches; no bolt preload, contact, "
           "fatigue, creep or moisture. Datasheet properties (Bambu PAHT-CF, PETG-CF), dry. The before parts carry a different load path "
           "(today's stage 1 is loaded at the roll horn only), so before -> after compares two designs under the same hand loads, not "
           "the same patches. Nothing printed or loaded.")
    fem = dict(id="fem", title="Arm load envelope FEM, before -> after (recovery case)", kind="simulation",
               observation=f"Over the full arm load envelope both U connectors keep every wrench above strength and layer ratio 1 in both "
                           f"materials; today's stages fall below strength ratio 1 for {below} of the wrenches of the 4 kg cases. Recovery case, PAHT-CF: blue U worst "
                           f"strength ratio {rows[(S1, MATS[0], 'recovery_4kg_1p5g')]['after']['p99_strength_ratio']['min']:.2f} (median "
                           f"{rows[(S1, MATS[0], 'recovery_4kg_1p5g')]['after']['p99_strength_ratio']['p50']:.2f}), green U "
                           f"{rows[(S2, MATS[0], 'recovery_4kg_1p5g')]['after']['p99_strength_ratio']['min']:.2f} (median "
                           f"{rows[(S2, MATS[0], 'recovery_4kg_1p5g')]['after']['p99_strength_ratio']['p50']:.2f}); peak displacement "
                           f"{rows[(S1, MATS[0], 'recovery_4kg_1p5g')]['after']['max_displacement_mm']['max']:.2f} / "
                           f"{rows[(S2, MATS[0], 'recovery_4kg_1p5g')]['after']['max_displacement_mm']['max']:.2f} mm (today "
                           f"{rows[(S1, MATS[0], 'recovery_4kg_1p5g')]['before']['max_displacement_mm']['max']:.0f} / "
                           f"{rows[(S2, MATS[0], 'recovery_4kg_1p5g')]['before']['max_displacement_mm']['max']:.0f} mm). The green U, "
                           "PETG-CF, recovery has the smallest margin. Values are the worst wrench with its median; the chart shows the whole distribution.",
               measurements=rec, conditions=cond, limitations=lim, source_refs=refs,
               image_groups=[{"title": "Distribution over the envelope, per part, material and case", "images": [
                   {"path": rel(CHART), "caption": "Right side. Thin line min-max, bar p10-p90, gap p50 of the 8310 wrenches; orange today, blue the U connectors."}]}])
    fem_cases = dict(id="fem-other-cases", title="Arm load envelope FEM, support and manipulation cases", kind="simulation",
                     observation="The two lighter cases scale with the recovery case; no wrench below ratio 1 for either U in either material.",
                     measurements=other, conditions=cond, limitations=lim, source_refs=refs)
    fem_mesh = dict(id="fem-mesh-and-left", title="FEM mesh convergence and left side", kind="simulation",
                    observation="Refining the mesh one level changes the worst p99 stress by a few percent; the mirrored left parts give the "
                                "same distributions as the right (their envelope has 8383 wrenches: the left arm's own pose grid).",
                    measurements=conv, conditions=dict(mesh=cond["mesh"], left="L parts = mirror x of the R parts, setups mirrored (x -> -x)"),
                    limitations=lim, source_refs=refs)
    pitch = dict(id="pitch-output", title="Pitch servo output load (still single-sided)", kind="analysis",
                 observation="The pitch servo is still held on its output side only (no pitch idler). Over the arm load envelope its output "
                             f"carries up to {pm['recovery_4kg_1p5g']['off_axis_Nm']['max']} N m off-axis (median "
                             f"{pm['recovery_4kg_1p5g']['off_axis_Nm']['p50']} N m, recovery) besides the torque about its axis, which the "
                             "envelope caps at the 4.9 N m stall. There is no STS3250 data for the off-axis load its output bearing takes.",
                 measurements=[m(f"Off-axis moment on the pitch output, {CASES[c].split(' (')[0]}", pm[c]["off_axis_Nm"]["max"], "N m",
                                 f"worst wrench; {qtext(pm[c]['off_axis_Nm'])}; about the axis max {pm[c]['axial_Nm']['max']} N m")
                               for c in CASES],
                 conditions={"method": "the envelope wrenches (arm_loads.envelope) taken at the R_shoulder_pitch joint origin and split into "
                                       "the component along the pitch axis and the rest", "file": rel(MOMENTS)},
                 limitations="Kinematic load transfer only; how the servo's bearing and case share the off-axis moment is unknown.",
                 source_refs=[{"label": "Pitch output moments", "path": rel(MOMENTS)}])
    return [fem, fem_cases, fem_mesh, pitch]


def is_shell(*names):
    return any(n in SHELLS for n in names)


def clearance_evidence(refs):
    out = {}
    for side, label in (("R", "shoulder-u-verify3"), ("L", "shoulder-u-verify-L")):
        ch = report(label)["checks"]
        clr = ch["clearance"]["per_part"]
        near = sorted(((v["min_distance_mm"], a, b, v["pose_of_min"]) for a, p in clr.items() for b, v in p.items() if v["min_distance_mm"] < 5.0))
        sweep = []
        for jn, j in ch["range_sweep"]["per_joint"].items():
            lo, hi = j["assembly_range_deg"]
            for pr in j["pairs"]:
                inside = [(d, v) for d, v in pr["angles"] if lo <= d <= hi]
                sweep.append(dict(joint=jn, a=pr["a"], b=pr["b"], range=(lo, hi), shell=is_shell(pr["a"], pr["b"]),
                                  first=[d for d, _ in pr["angles"]], inside_max=max((v for _, v in inside), default=0.0),
                                  inside_from=[d for d, _ in inside], overall_max=max(v for _, v in pr["angles"])))
        stand = [(w["part"], w["mating_part"], w["value"]) for w in ch["static_audit"]["worst_cases"]]
        gaps = {(w["part"], w["mating_part"]): (w["value"], w["overlap_mm3"]) for w in ch["gaps"]["worst_cases"]}
        out[side] = dict(near=near, sweep=sweep, stand=stand, gaps=gaps, poses=ch["clearance"]["notes"].split(",")[0],
                         new_pairs=ch["clearance"]["bad_events"]["pairs_with_new_overlap"])
    R = out["R"]
    other_sweep = [s for s in R["sweep"] if not s["shell"]]
    shell_sweep = [s for s in R["sweep"] if s["shell"]]
    g = R["gaps"]
    pick = lambda a, b: g.get((a, b), (None, None))[0]
    rest = [m("New overlap over the arm poses, neighbours on other links", 0.0 if R["new_pairs"] == 0 else None, "mm3",
              f"{R['poses']}; pairs with a new overlap: R {R['new_pairs']}, L {out['L']['new_pairs']}")]
    rest += [m(f"Closest approach, {a.replace('R_', '')} to {b.replace('upper__R_', '')}", d, "mm", pose) for d, a, b, pose in R["near"]]
    rest += [m(f"Range sweep {s['joint']}: {s['a']} x {s['b']}", s["inside_max"], "mm3 overlap inside the joint range",
               f"range {s['range'][0]:g}..{s['range'][1]:g} deg; contact only at {min(s['first']):g}..{max(s['first']):g} deg, max {s['overall_max']} mm3")
             for s in other_sweep]
    rest += [m(f"Standing gap, {S1.split('_combined')[0]} to roll main horn / idler", pick("R_" + S1, "upper__R_roll_C047_main_horn"), "mm",
               f"idler {pick('R_' + S1, 'upper__R_roll_C047_rear_idler')} mm, pitch main horn {pick('R_' + S1, 'upper__R_pitch_C047_main_horn')} mm (seated faces)"),
             m(f"Standing gap, {S2.split('_combined')[0]} to roll case / yaw case", pick("R_" + S2, "upper__R_roll_C047_case"), "mm",
               f"yaw case {pick('R_' + S2, 'upper__R_yaw_C047_case')} mm (seated faces)"),
             m("Part vs its own sleeves at standing (press interference)", g.get(("R_" + S1, "R_" + S1 + "_steel_sleeves"), (0, 0))[1], "mm3",
               f"green {g.get(('R_' + S2, 'R_' + S2 + '_steel_sleeves'), (0, 0))[1]} mm3")]
    other = dict(id="clearance-mechanism", title="Clearance and range: servos, horns, idler, other connectors", kind="analysis",
                 observation="Against everything except the torso shells and the upper-arm cover, the two U connectors add no overlap over "
                             "the arm poses and none inside the pitch and roll ranges. Closest approach: blue U to the roll servo case "
                             f"{R['near'][0][0]} mm at the pitch +150 / roll -15 corner, blue U to the pitch servo case {R['near'][1][0]} mm. "
                             "The only mechanism contact in the sweep is the roll case touching the blue U at roll -30 deg, 15 deg past the "
                             "range end. Left side the same to 0.01 mm.",
                 measurements=rest,
                 conditions={"poses": R["poses"] + " (each arm joint at both range ends, named combinations, chain corners)",
                             "range_sweep": "2.5 deg steps over the swept span, pitch -90..180, roll -30..120; ranges pitch -60..150, roll -15..95",
                             "method": "mesh booleans of closed screening solids (native STEP for the scoped parts), distance capped at 5 mm"},
                 limitations="Rigid kinematic sweep; no compliance, no cable. The generated screws are not in the swept set.",
                 source_refs=refs)
    shell = dict(id="clearance-shells", title="Torso shells and upper-arm cover: overlaps deferred to the shell redesign", kind="analysis",
                 observation="The torso centre shells and the upper-arm cover were drawn around today's connectors; the U connectors pass "
                             "through them. These overlaps are left for the shell redesign (Idle T-tph7xb) and are listed here, not fixed.",
                 measurements=[m(f"Standing overlap, {a} x {b}", v, "mm3") for a, b, v in R["stand"]]
                              + [m(f"Range sweep {s['joint']}: {s['a']} x {s['b']}", s["inside_max"], "mm3 max overlap inside the joint range",
                                   f"range {s['range'][0]:g}..{s['range'][1]:g} deg; overlapping at {len(s['inside_from'])} of the 2.5 deg steps inside it; "
                                   f"max over the swept span {s['overall_max']} mm3") for s in shell_sweep],
                 conditions={"parts": "center_front_shell, center_back_shell, R/L_upper_arm_hard_cover (today's shells, unchanged)"},
                 limitations="The shells are mesh-only; their screening solids are approximate. The upper-arm cover also lost its two "
                             "posts on the green U (dropped with their M3 screws and nuts): it has no mount on the new parts yet.",
                 source_refs=refs)
    return [other, shell]


def print_evidence(refs):
    rows, per = [], {}
    for side, label in (("R", "shoulder-u-verify3"), ("L", "shoulder-u-verify-L")):
        ch = report(label)["checks"]
        for base in (S1, S2):
            n = side + "_" + base
            wl, ov, bd, ho, pr, ge = (ch[c]["per_part"][n] for c in ("walls", "overhang", "bed", "holes", "press", "geometry"))
            boss = [h for h in ho["hole_list"] if h.get("boss_wall_mm") is not None]
            thin = min(boss, key=lambda h: h["boss_wall_mm"])
            sup = [s["counter_face"]["support_pct"] for s in pr["sleeves"]]
            per[n] = dict(up=ov["up"], wl=wl, ov=ov, bd=bd, ho=ho, pr=pr, ge=ge, thin=thin, sup=sup,
                          horiz=ho["horizontal_holes_with_hardware"], lw=wl["line_width_mm"])
    for base in (S1, S2):
        p = per["R_" + base]; pl = per["L_" + base]; lab = NAMES[base].split(" (")[0]
        up = p["up"] if isinstance(p["up"], str) else "plate-down (yaw-local normal, 9.5 deg off +z)"
        rows += [m(f"{lab}: print direction", up, "", f"L {pl['up'] if isinstance(pl['up'], str) else 'mirrored'}; Bambu X2D, PAHT-CF, 0.6 mm nozzle, 0.3 mm layers"),
                 m(f"{lab}: bed contact", p["bd"]["contact_mm2"], "mm2", f"{p['bd']['islands']} island, {round(100 * p['bd']['contact_share_of_footprint'])} % of the "
                   f"footprint, centre of mass {p['bd']['com_inside_contact_hull_mm']} mm inside the contact, {p['bd']['sharp_corners']} outline corners under 100 deg"),
                 m(f"{lab}: thinnest wall", p["wl"]["wall_min_mm"], "mm", f"line width {p['lw']} mm; {p['wl']['wall_under_2_lines_mm2']} mm2 under 2 lines, "
                   f"0 under 1 line; median {p['wl']['wall_median_mm']} mm"),
                 m(f"{lab}: downward area beyond the 70 deg overhang limit", p["ov"]["beyond_limit_mm2"], "mm2",
                   f"of {p['ov']['part_surface_mm2']} mm2; widest bridge {p['ov']['longest_bridge_span_mm']} mm (limit {p['ov']['max_bridge_mm']}); "
                   f"{p['ov']['floating_ceilings']} small floating ceilings (sleeve-bore tops); on mating faces {p['ov']['beyond_limit_on_mating_faces_mm2']} mm2"),
                 m(f"{lab}: thinnest boss wall around a hardware hole", p["thin"]["boss_wall_mm"], "mm",
                   f"{p['thin'].get('role')} d {p['thin']['d_mm']} mm" + (f" for {p['thin']['screw']}" if p["thin"].get("screw") else "")
                   + f"; L {pl['thin']['boss_wall_mm']} mm; {p['horiz']} hardware holes lie across the layers (sag unless teardropped or reamed)"),
                 m(f"{lab}: sleeve press, spacer-block support, worst sleeve", min(p["sup"]), "%",
                   f"{len(p['sup'])} sleeves, tool-path obstruction {p['pr']['summary']['max_tool_obstruction_mm3']} mm3; "
                   f"CAD press interference 0.0 mm, printer hole offset not measured"),
                 m(f"{lab}: B-rep events", p["ge"]["micro_edges"] + p["ge"]["sliver_faces"] + p["ge"]["acute_wedge_edges"] + p["ge"]["invalid_solids"], "events",
                   f"invalid {p['ge']['invalid_solids']}, slivers {p['ge']['sliver_faces']}, micro-edges {p['ge']['micro_edges']} (shortest "
                   f"{p['ge']['min_edge_length_mm']} mm), acute wedges {p['ge']['acute_wedge_edges']}, unrounded profile corners "
                   f"{p['ge']['unrounded_profile_corners']}; L/R volume difference {p['ge']['mirror_volume_diff_pct']} %; build-volume margin "
                   f"{p['ge']['min_bed_margin_mm']} mm")]
    s2 = per["R_" + S2]
    return [dict(id="print", title="Print setup and CAD rules per part", kind="analysis",
                 observation="Blue U printed on its pitch-hub face (R +x, L -x): the only direction with a flat bed face and every bridge "
                             f"under 40 mm (widest {per['R_' + S1]['ov']['longest_bridge_span_mm']} mm, under the bar). Green U plate-down on "
                             f"the bed ({s2['bd']['contact_mm2']} mm2 contact). Walls stay at or above two line widths except "
                             f"{per['R_' + S1]['wl']['wall_under_2_lines_mm2']} mm2 on the blue idler arm. On the right green U the wall "
                             f"around one yaw-case PA2 clearance hole (today's yaw interface) is {s2['thin']['boss_wall_mm']} mm thin; on the left "
                             f"it is {per['L_' + S2]['thin']['boss_wall_mm']} mm, so the mirrored sides differ there.",
                 measurements=rows,
                 conditions={"checks": "walls, overhang, bed, holes, press, geometry of robot/body/checks/run.py (print settings from part_settings.json)"},
                 limitations="Geometry rules on the STEP, not a slice; support volume and print time not run (slicer check). The geometry "
                             "check's load-path slab setting (loadpath_slices) is still the one of today's connectors, so its slab areas "
                             "do not describe these parts and are left out.",
                 source_refs=refs)]


def screw_evidence(refs):
    man = json.loads(MANIFEST.read_text())
    gen = [p for p in man["parts"] if p["name"].startswith("R_shoulder_u_")]
    groups = {}
    for p in gen:
        sp = p["screw_plan"]; key = f"{sp['thread_label']}x{sp['length_mm']:g} into {sp['hole_id'].split(':')[0].replace('upper__R_', '')}"
        groups.setdefault(key, []).append(sp)
    rows = [m(f"Generated {k}: thread engagement", min(s["engagement_mm"] for s in v), "mm",
              f"{len(v)} per side, ISO 7380-1; reference {v[0]['reference_engagement_mm']} mm (hole depth {v[0]['hole_depth_mm']}); "
              f"head gap {min(s['head_gap_mm'] for s in v)}..{max(s['head_gap_mm'] for s in v)} mm; tip margin {min(s['tip_margin_mm'] for s in v)} mm")
            for k, v in groups.items()]
    ch = report("shoulder-u-verify3")["checks"]
    eng = [w for w in ch["screws"]["worst_cases"] if w["unit"] == "mm engagement shortfall"]
    yaw = [w for w in eng if "yaw" in w["what"]]
    pit = [w for w in eng if "pitch_horn" in w["what"]]
    rows += [m("Kept yaw-case PA2 screws (today's interface): thread engagement", round(3.0 - max(w["value"] for w in yaw), 3), "mm",
               f"{len(yaw)} screws, reference 3.0 mm, shortfall up to {max(w['value'] for w in yaw)} mm"),
             m("Kept pitch-horn M3x6: thread engagement", round(2.5 - max(w["value"] for w in pit), 3) if pit else None, "mm",
               "4 screws, reference 2.5 mm; the washer model overlaps the pitch sleeves by 1 mm (stale hardware model of the kept interface)")]
    da = ch["driver_access"]["per_part"]
    for base in (S1, S2):
        for n, s in da["R_" + base]["screws"].items():
            rows.append(m(f"Driver path of {n.replace('upper__R_', '').replace('R_', '')}: first obstruction from the head", s["first_distance_mm"], "mm",
                          f"{s['first_part']}; {s['obstruction_mm3']} mm3 = {s['blocked_share_pct']} % of the {s['path_length_mm']} mm blade path"))
    return [dict(id="screws", title="Screws and assembly access", kind="analysis",
                 observation="The 12 new screws per side are generated from the STS3250's defined holes: horn M3x8 at the full 2.5 mm "
                             "engagement, idler M3x6 at 1.9 of 2.1 mm, case M2x12 at 3.1-3.3 of 3.0 mm. The kept yaw-case PA2 screws "
                             "engage 1.8 of 3.0 mm (today's interface, unchanged). Assembly order: the roll servo case closes the driver "
                             "path of the four pitch-horn screws (7-12 mm from the head) and of the two inner yaw-case PA2x7 (at the head), "
                             "so those are driven before the roll servo goes in.",
                 measurements=rows,
                 conditions={"generated": "robot/body/fastening/screw_plan.py from the defined holes (manifest screw_plan rows)",
                             "existing_screws": "screws and driver_access checks of run.py on the kept pitch-horn and yaw-case screws (blade 3.0 mm, assumption)"},
                 limitations="The shared screws and driver_access checks did not pick up the generated screw meshes (they count the 12 new "
                             "sleeves as sleeves without a screw); the generated screws' numbers come from the screw plan, and their driver "
                             "paths are not measured. No assembly-order data (assembly_order.json) for the shoulder yet.",
                 source_refs=refs)]


def mass_bom_evidence(refs):
    man = json.loads(MANIFEST.read_text())
    by = {p["name"]: p for p in man["parts"]}
    before = {p["name"]: p for p in json.loads((ROOT / man["shoulder_u"]["source_manifest"]).read_text())["parts"]}
    rho = json.loads((ROOT / "robot/body/checks/data/materials.json").read_text())["Bambu PAHT-CF"]["density_g_cm3"]
    g = lambda n: round(by[n]["mass_kg"] * 1000, 2)
    gb = lambda n: round(before[n]["volume_mm3"] * rho / 1000, 2)
    gs = lambda n: round(before[n]["mass_kg"] * 1000, 2)
    screws = sum(p["mass_kg"] for p in man["parts"] if p["name"].startswith("R_shoulder_u_")) * 1000
    added = g("R_" + S1) - gb("R_" + S1) + g("R_" + S2) - gb("R_" + S2) + g("R_" + S1 + "_steel_sleeves") - gs("R_" + S1 + "_steel_sleeves") \
        + g("R_" + S2 + "_steel_sleeves") - gs("R_" + S2 + "_steel_sleeves")
    ml = report("shoulder-u-verify3")["checks"]["mass"]["per_link"]
    mass = dict(id="mass", title="Mass", kind="measurement",
                observation=f"Per side the two U connectors weigh {g('R_' + S1)} g and {g('R_' + S2)} g, against {gb('R_' + S1)} g and "
                            f"{gb('R_' + S2)} g for today's stages (same density): the printed parts and sleeves add {added:.1f} g per side, "
                            f"plus {screws:.1f} g of new screws (the removed hardware is not weighed in the manifest).",
                measurements=[m("Blue U", g("R_" + S1), "g", f"PAHT-CF {rho} g/cm3, per side; today {gb('R_' + S1)} g"),
                              m("Blue U sleeves", g("R_" + S1 + "_steel_sleeves"), "g", f"12 x 304 SS; today {gs('R_' + S1 + '_steel_sleeves')} g (8)"),
                              m("Green U", g("R_" + S2), "g", f"per side; today {gb('R_' + S2)} g"),
                              m("Green U sleeves", g("R_" + S2 + "_steel_sleeves"), "g", f"8 x 304 SS; today {gs('R_' + S2 + '_steel_sleeves')} g (8)"),
                              m("New screws", round(screws, 2), "g", "12 per side, mesh volume x 7.85 g/cm3"),
                              m("Pitch link (parts with a mass)", ml["R_shoulder_pitch_link"]["mass_g"], "g",
                                f"{ml['R_shoulder_pitch_link']['parts_without_mass']} parts without a mass (horns, idler, hardware)"),
                              m("Roll link (parts with a mass)", ml["R_shoulder_roll_link"]["mass_g"], "g",
                                f"{ml['R_shoulder_roll_link']['parts_without_mass']} parts without a mass")],
                conditions={"basis": "native STEP volume x part-setting material density; before: today's STEP volume at the same density "
                                     "(its manifest mass used the old 1.22 g/cm3)"},
                limitations="The roll idler's mass is inside the servo's catalogue mass. Link masses leave out parts without a mass.",
                source_refs=refs)
    ch = man["shoulder_u"]["changes"]
    cnt = lambda pat: sum(1 for c in ch if c.startswith("upper__R_" + pat) or c.startswith("R_" + pat))
    removed = [("4 PA2x7, 4 washers, 4 spacers on the roll case -y face (today's case screws)", cnt("roll_case_")),
               ("4 M3x6 + 4 washers on the roll horn (replaced by generated M3x8)", cnt("roll_horn_")),
               ("2 cover-post M3x8 + 2 M3 nuts (the two cover posts on stage 2 are dropped)", cnt("upper_arm_shell_"))]
    bom = dict(id="bom", title="Bill of materials changes", kind="analysis",
               observation="Per side: one new roll rear idler (the STS3250's idler member); 4 M3x8 + 4 M3x6 + 4 M2x12 ISO 7380-1 button "
                           "screws generated from the servo's defined holes; 4 more M3 sleeves on the blue U and 4 new M2 sleeves on the "
                           "green U. Dropped: today's 4 PA2 with washers and spacers on the roll case -y face, the roll-horn M3x6 with "
                           "washers, and the two cover posts with their M3 screws and nuts.",
               measurements=[m("Roll rear idler, added", 1, "per side", "STS3250 idler member, placed with the registered roll case transform")]
                            + [m(f"Generated {k}", len(v), "per side") for k, v in
                               {f"{p['screw_plan']['thread_label']}x{p['screw_plan']['length_mm']:g} ISO 7380-1, "
                                f"{p['screw_plan']['hole_id'].split('.')[-2]}": [q for q in man['parts'] if q['name'].startswith('R_shoulder_u_')
                                                                                 and q['screw_plan']['hole_id'].split('.')[-2] == p['screw_plan']['hole_id'].split('.')[-2]]
                                for p in man["parts"] if p["name"].startswith("R_shoulder_u_")}.items()]
                            + [m(f"Removed: {t}", n, "parts per side") for t, n in removed]
                            + [m("Steel sleeves, blue U", by["R_" + S1 + "_steel_sleeves"]["native_solid_count"], "per side",
                                 f"today {before['R_' + S1 + '_steel_sleeves'].get('native_solid_count')}: 4 pitch (4.5 x 3.3) kept + 8 M3 (4.5 x 3.2) at horn and idler"),
                               m("Steel sleeves, green U", by["R_" + S2 + "_steel_sleeves"]["native_solid_count"], "per side",
                                 f"today {before['R_' + S2 + '_steel_sleeves'].get('native_solid_count')}: 4 yaw-saddle (3.5 x 2.4) kept + 4 M2 (3.6 x 2.2) at the case holes")],
               conditions={"source": "shoulder_u.changes in the candidate manifest (shoulder_clevis.py manifest)"},
               limitations="No buy lines, suppliers or prices: the robot BOM (robot/body/bom) is generated from the selected manifest and "
                           "is not regenerated for this candidate.",
               source_refs=[{"label": "Candidate manifest", "path": rel(MANIFEST)}])
    return [mass, bom]


def record():
    man = json.loads(MANIFEST.read_text())
    refs = [{"label": f"Check report {k} ({v})", "path": rel(ART / "checks" / f"{k}.json")} for k, v in REPORTS.items()]
    evidence = fem_evidence(refs[2:]) + clearance_evidence(refs[:2]) + print_evidence(refs[:2]) + screw_evidence(refs[:2]) + mass_bom_evidence(refs[:1])
    renders = [ART / "renders" / n for n in RENDERS] + [CHART]
    part_files = sorted(f for s in ("R_", "L_") for n in (S1, S2) for f in (ART / "parts").glob(f"{s}{n}*.step")) \
        + sorted((ART / "parts").glob(f"upper__?_{IDLER}.step"))
    data = [MANIFEST, ART / "cad-manifest.json", ART / "connectors-bilateral.step", MOMENTS,
            ROOT / "robot/body/checks/data/part_settings.json"] + [ART / "checks" / f"{k}.json" for k in REPORTS] + part_files

    def slug(p):
        return Path(p).name.replace(".", "-").replace("_", "-").lower()
    assets = ([{"id": "neutral", "title": "Complete body with the shoulder U connectors", "kind": "model", "path": rel(ART / "assembly-shoulder-u.glb")}]
              + [{"id": slug(p), "title": p.stem.replace("-", " "), "kind": "image", "path": rel(p)} for p in renders]
              + [{"id": slug(p), "title": p.name, "kind": "data", "path": rel(p)} for p in data])
    fem = evidence[0]["measurements"]
    title = "Shoulder roll held on both faces (blue and green U)"
    share = [100.0 * r["before"]["wrenches_with_p99_strength_ratio_below_1"] / r["before"]["wrenches"]
             for (pt, mt, c), r in fem_rows().items() if c != "manip_1p5kg_reach"]
    below = f"{min(share):.0f}-{max(share):.0f} %"
    pm = json.loads(MOMENTS.read_text())["cases"]["recovery_4kg_1p5g"]
    provenance = ("Idle T-ce88t9 (shoulder), T-ysng9c (arm load envelope), T-2e49mv (this build and publication); decisions M-s8gt2p (load "
                  "envelope), M-e9bjtf / M-s9tnc9 (the owner's U sketch), M-rya3ty (owner 2026-11-20: \"Much better the two U connectors look "
                  "clean now. Continue\"); commits 31979a94b (first build), 37df2fd4c (FEM setup, blue convergence), 6b10b68b4 (green stiffened, "
                  "blue hub, print directions), and this publisher.")
    limitations = ("Torso shell and upper-arm cover overlaps deferred to the shell redesign (Idle T-tph7xb); the cover has no mount on the "
                   "new parts yet. Yaw-case PA2 screws engage 1.8 of 3.0 mm (today's interface, unchanged). Assembly order: the pitch-horn "
                   "screws and the inner yaw-case screws must be driven before the roll servo goes in (driver_access). The notch in the "
                   "green U's front arm (front view, about x 130 mm) is still open. The pitch servo is still held on one side only: its "
                   f"output carries up to {pm['off_axis_Nm']['max']} N m off-axis in the envelope (recovery) and there is no servo data for that. The candidate body "
                   f"was derived from the selected manifest at sha256 {man['shoulder_u']['source_manifest_sha256'][:12]} (before the blue "
                   "thigh shell), so its legs show that earlier state; this study owns the shoulder only. Robot model (URDF) and controller "
                   "not updated. Nothing printed or loaded.")
    rec = {
        "id": STUDY, "study_id": STUDY, "title": title,
        "source": [rel(HERE / "shoulder_clevis.py"), rel(Path(__file__))],
        "dashboard": "design", "workspace": "Body", "robot_sections": ["Shoulder"],
        "observation": "The shoulder roll servo is now held on both faces. The blue U on the pitch link grips the roll horn and a new "
                       "roll idler; the green U on the roll link clamps the roll case on its 2 + 2 far holes and keeps today's yaw clamp. "
                       "Over the arm load envelope (8310 hand-force wrenches per case) no wrench takes either part below strength or "
                       f"layer ratio 1. Today's stages go below strength ratio 1 for {below} of the wrenches of the 4 kg cases. The parts are clear of the servos, "
                       "horns, idler and each other over the pitch and roll ranges. The torso shells and the upper-arm cover still overlap; "
                       "they are left for the shell redesign.",
        "measurements": "; ".join(f"{x['name']}: {x['value']} {x['unit']}" + (f" ({x['condition']})" if x.get("condition") else "")
                                  for x in (fem[0], fem[2], fem[10], fem[12])),
        "limitations": limitations,
        "provenance": [provenance],
        "assets": assets, "clips": [],
        "design": {
            "id": STUDY, "title": title,
            "summary": "Roll servo held on both faces: blue U (pitch link) on roll horn + new idler, green U (roll link) on the 2 + 2 far case "
                       "holes. Recovery envelope, PAHT-CF: worst strength ratio "
                       f"{fem[0]['value']} / {fem[10]['value']} (today about 0.1-0.3). Shells and cover overlaps deferred.",
            "visual_kind": "native_CAD_assembly",
            "maturity": "CAD with the shared checks on both sides (fit, motion, print rules, screws, envelope FEM with mesh convergence); "
                        "owner accepted the look (M-rya3ty). Not printed, not loaded, model not updated.",
            "image_path": rel(ART / "renders" / "shoulder-u-parts.png"),
            "cad": {
                "model_units": "m",
                "views": [{"label": "Complete body with the shoulder U connectors", "model_path": rel(ART / "assembly-shoulder-u.glb")}],
                "parts_path": rel(MANIFEST),
                "step_path": rel(ART / "connectors-bilateral.step"),
                "report_path": rel(ART / "checks" / "shoulder-u-verify3.json"),
                "hide_shell_toggle": True, "show_shells": True,
                "motion_note": "Joint sliders are inspection commands over the manifest ranges, not qualified travel.",
            },
            "source_refs": refs + [{"label": "Candidate manifest", "path": rel(MANIFEST)}, {"label": "CAD build record", "path": rel(ART / "cad-manifest.json")}],
            "evidence": evidence,
            "provenance": provenance,
        },
    }
    out = ART / "publication.json"
    out.write_text(json.dumps(rec, indent=1))
    print(out)


if __name__ == "__main__":
    {"package": package, "record": record}[sys.argv[1]]()
