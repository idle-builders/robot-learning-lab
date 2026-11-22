"""Published FEM evidence of a design study, recreated from the shared check's reports (no study-local FEM script).

  .venv/bin/python robot/body/checks/fem_evidence.py --study cove50-arm-fdm-combined --command   # prints the run.py command that produces the report
  .venv/bin/python robot/body/checks/fem_evidence.py --study cove50-arm-fdm-combined --report .cache/work/cad-speed/checks/<label>/report.json

The report comes from `run.py --parts <the study's parts> --checks fem,fem_setup` (--command prints it). The output is `fem-evidence.json` in the study's artifact
folder (artifacts/robot/body/<folder>/, outside Git: derived, recreated by this script): per part the layer direction, mesh, origin, patches, and per layer normal /
material / load case the p99 and peak von Mises, p99 layer tension, strength ratios and peak displacement; per chain (one side of the arm, neck, torso) the RSS of the
stage displacements per load case and material. The FEM section of the published records (robot/body/{arm,neck-head,torso}/publish_*_studies.py)
is built from this file through the helpers below (load, own_row, rss, min_ratio, stage_text, materials, render_chart); nothing else carries FEM numbers into a record.
The fit and CAD-rule sections of the records come from the same tool's clearance and geometry reports (check_command / load_check / check_cards below), recreated by
`fem_evidence.py --study S --check-command`. Materials: only those that the study's setups ran appear (materials(ev)); a material that was not run is never claimed. Not recreated, because the tool has
no counterpart: the old multi-family tables (baseline metal / truss / topopt; those families are not in the tree), the wet-conditioned PAHT-CF material and the Tsai-Hill
index (the checks report the in-plane p99 strength ratio instead), and the baseline-metal chain comparison. Parts without a setup have no result; `coverage` names why.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# study id -> artifact folder, parts the FEM runs on (run.py --parts globs), chains (name -> part-name prefix filter) for the RSS displacement
STUDIES = {
    "cove50-arm-fdm-combined": dict(folder="arm", parts=["?_shoulder_pitch_to_roll_combined", "?_shoulder_roll_to_yaw_combined", "?_upper_arm_yaw_to_elbow_combined",
                                                         "?_forearm_elbow_to_wrist_combined"], chains={"R": "R_", "L": "L_"}),
    "cove50-neck-head-fdm-combined": dict(folder="neck-head", parts=["combined__neck_*", "combined__head_sensor_compute_carrier"], chains={"neck": "combined__"}),
    "cove50-torso-fdm-combined": dict(folder="torso", parts=["combined__stage?_*"], chains={"torso": "combined__stage"},
                                    note="Only stage 1 (the chassis) has an FEM setup. Stage 2 has a setup but no native STEP in the manifest (mesh only), so it is not run. Stages 3 and 4 have no "
                                         "setup: the old torso study loaded them with a force and moment table (fem_evaluate LOAD_CASES) at hand-typed bore spheres that no longer match the "
                                         "geometry (the old sites sit about 12 mm off today's mating holes), so a setup needs its supports and load patches re-derived from the mating parts "
                                         "first; the old table alone would not be today's load path. The chain displacement below is therefore stage 1 alone, not a four-stage RSS."),
}


def command(study, label=None):
    """The run that recreates a study's FEM report. 4 h: the arm's hand-load envelope (8310 wrenches per case, 3 materials) took 3030 s
    (R stages 1-4, quiet machine: 338 + 771 + 1924) to 5250 s (R, 4 parallel jobs; stage 4 alone 4103 s) for one side, so about
    6000 s for both sides on a quiet machine and about 10500 s under the allowed parallel load."""
    s = STUDIES[study]
    label = label or study
    return (f"python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 14400 -- .venv/bin/python -B robot/body/checks/run.py --parts '{",".join(s["parts"])}' "
            f"--checks fem,fem_setup --label {label}")


def check_command(study):
    """The light run that recreates the geometry (CAD-rule) and clearance (fit) reports of a study's record."""
    return (f"python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 900 -- .venv/bin/python -B robot/body/checks/run.py --parts '{",".join(STUDIES[study]["parts"])}' "
            f"--checks geometry,clearance --label {study}-geometry")


def build(study, report_path):
    s = STUDIES[study]
    rep = json.loads(Path(report_path).read_text())
    fem = rep["checks"]["fem"]["per_part"]
    setup = (rep["checks"].get("fem_setup") or {}).get("per_part", {})
    parts = {}
    for n, p in sorted(fem.items()):
        rows = [dict(layer_normal=ln, material=m, load_case=c, **{k: v[k] for k in ("p99_vm_MPa", "peak_vm_MPa", "p99_layer_tension_MPa", "p99_strength_ratio", "p99_layer_ratio",
                                                                                    "max_displacement_mm", "peak_vm_location_mm")})
                for ln, bm in p["by_layer_normal"].items() for m, cs in bm.items() for c, v in cs.items()]
        parts[n] = dict(layer_direction=p["layer_direction"], nodes=p["nodes"], mesh_h_mm=p["mesh_h_mm"], origin_mm=p["origin_mm"], rigid_load=p["rigid_load"],
                        wrenches_N_Nmm=p["wrenches_N_Nmm"], reaction_plus_wrench_pct_max=p["reaction_plus_wrench_pct_max"],
                        patches={k: {q: v[q] for q in ("area_mm2", "nodes", "centroid_mm")} for k, v in p["patches"].items()},
                        setup_check={k: v for k, v in (setup.get(n, {}).get("support_load_overlap") or {}).items()},
                        open_difference=p["setup_notes"].get("open_difference"), rows=rows)
    chains = {}
    for cn, prefix in s["chains"].items():
        members = [n for n in parts if n.startswith(prefix)]
        own = {n: [r for r in parts[n]["rows"] if r["layer_normal"] == parts[n]["layer_direction"].get("print_axis", r["layer_normal"])] for n in members}   # each part along its own print direction
        keys = sorted({(r["material"], r["load_case"]) for n in members for r in own[n]})
        chains[cn] = dict(parts=members, rss_displacement_mm={"/".join(k): round(math.sqrt(sum(r["max_displacement_mm"] ** 2 for n in members for r in own[n]
                                                                                              if (r["material"], r["load_case"]) == k)), 4) for k in keys})
    cov = (rep["checks"].get("fem_setup") or {}).get("coverage", {})
    why = {n: ("setup not run: " + cov.get("setup_not_checked", {})[n]) if n in cov.get("setup_not_checked", {}) else
           "no fem setup for this printed part" if n in cov.get("printed_without_setup", []) else "no result in the report"
           for n in rep.get("scope", {}).get("parts", []) if n not in fem}
    return dict(schema="robot-learning-lab.fem-evidence.v1", study=study, note=s.get("note"), coverage=why, report=str(Path(report_path)), manifest=rep["manifest"], manifest_sha256=rep["manifest_sha256"],
                parts_without_result=[n for n in rep.get("scope", {}).get("parts", []) if n not in fem], parts=parts, chains=chains)


# helpers for the publishers: the FEM section of a record comes from fem-evidence.json only
MATERIALS = dict(dry="Bambu PAHT-CF", petg="Bambu PETG-CF")   # datasheet entries of checks/data/materials.json; no wet-conditioned PAHT-CF there


def materials(ev):
    """Datasheet materials (MATERIALS values) that the study's setups ran, in MATERIALS order. Only these may be named in a record."""
    ran = {r["material"] for p in ev["parts"].values() for r in p["rows"]}
    return [m for m in MATERIALS.values() if m in ran]


def chain_text(ev, chain, case):
    """'x mm (Bambu PAHT-CF) / y mm (...)' for the materials that ran; a material that did not run is skipped, never printed."""
    return " / ".join(f"{v:.3f} mm ({m})" for m in materials(ev) if (v := rss(ev, chain, m, case)) is not None)


def ratio_text(ev, parts, case):
    return " / ".join(f"{v:.2f}x {m}" for m in materials(ev) if (v := min_ratio(ev, parts, m, case)) is not None)


def load(study):
    path = ROOT / "artifacts/robot/body" / STUDIES[study]["folder"] / "fem-evidence.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `{command(study)}` and then fem_evidence.py --study {study} --report <that report.json>")
    return json.loads(path.read_text())


def own_row(ev, part, material, case):
    """The part's row along its own print direction, or None (no result)."""
    p = ev["parts"].get(part)
    if not p:
        return None
    ax = p["layer_direction"].get("print_axis")
    return next((r for r in p["rows"] if r["material"] == material and r["load_case"] == case and r["layer_normal"] == (ax or r["layer_normal"])), None)


def rss(ev, chain, material, case):
    return ev["chains"][chain]["rss_displacement_mm"].get(f"{material}/{case}")


def min_ratio(ev, parts, material, case):
    v = [r["p99_strength_ratio"] for r in (own_row(ev, n, material, case) for n in parts) if r]
    return min(v) if v else None


def stage_text(ev, part, material, case):
    r = own_row(ev, part, material, case)
    if not r:
        return f"no FEM result ({ev['coverage'].get(part, 'no result')})"
    return f"p99={r['p99_vm_MPa']:.2f} MPa, u_max={r['max_displacement_mm']:.3f} mm, {r['p99_strength_ratio']:.2f}x p99 strength ratio"


def stage_measurements(ev, label, part, material, case, what):
    r = own_row(ev, part, material, case)
    if not r:
        return []
    return [dict(name=f"{label} {what} p99 von Mises stress", value=r["p99_vm_MPa"], unit="MPa"),
            dict(name=f"{label} {what} max deflection", value=r["max_displacement_mm"], unit="mm"),
            dict(name=f"{label} {what} p99 strength ratio ({material})", value=r["p99_strength_ratio"], unit="x")]


def limitations(ev):
    s = (f"FEM numbers are the shared checks' static linear P1 tetrahedral results (fem-evidence.json, from run.py --checks fem,fem_setup), per part along its print direction, "
         f"materials run: {', '.join(materials(ev)) or 'none'} (datasheet; a material not named here was not run for this study). No counterpart in the new evidence: wet-conditioned PAHT-CF, Tsai-Hill index, the baseline-metal chain and the truss / topopt families. ")
    if ev.get("note"):
        s += ev["note"] + " "
    return s + "Physical bolt preload retention and cyclic fatigue need bench verification."


def render_chart(ev, out, parts, case, labels, title):
    """Stress / displacement / strength-ratio chart of the listed parts (each along its print direction) for one load case: drawn from fem-evidence.json, only the materials that ran."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mats = materials(ev)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    w = 0.8 / max(len(mats), 1)
    for ax, key, ylab in zip(axs, ("p99_vm_MPa", "max_displacement_mm", "p99_strength_ratio"), ("p99 von Mises (MPa)", "max displacement (mm)", "p99 in-plane strength ratio (x)")):
        for i, m in enumerate(mats):
            rows = [own_row(ev, n, m, case) for n in parts]
            ax.bar([k + i * w for k in range(len(parts))], [r[key] if r else 0 for r in rows], w, label=m)
            for k, r in enumerate(rows):
                if not r:
                    ax.text(k + i * w, 0, "no result", rotation=90, va="bottom", ha="center", fontsize=7)
        ax.set_xticks([k + w * (len(mats) - 1) / 2 for k in range(len(parts))]); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylab); ax.grid(axis="y", alpha=.3)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"{title}: {case}; shared-checks FEM (fem-evidence.json), materials run: {', '.join(mats)}", fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return Path(out)


def load_check(study):
    """The geometry + clearance report of the study (run.py --checks geometry,clearance), or exit with the command that recreates it."""
    path = ROOT / ".cache/work/cad-speed/checks" / f"{study}-geometry/report.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `{check_command(study)}`")
    return json.loads(path.read_text())


def check_numbers(rep):
    """Totals of the geometry and clearance reports; counts next to the measurements, no verdicts."""
    geo = rep["checks"]["geometry"]["per_part"]
    clr = rep["checks"]["clearance"]["per_part"]
    paths = [g["loadpath"]["min_slab_area_mm2"] for g in geo.values() if g.get("loadpath")]
    vals = [v for c in clr.values() for v in c.values()]
    return dict(parts=len(geo), sleeves=sum(g["sleeves"] for g in geo.values()),
                max_step_vs_mesh_pct=max(g["step_vs_mesh_volume_diff_pct"] for g in geo.values()),
                sliver_faces=sum(g["sliver_faces"] for g in geo.values()), micro_edges=sum(g["micro_edges"] for g in geo.values()),
                acute_wedge_edges=sum(g["acute_wedge_edges"] for g in geo.values()), unrounded_corners=sum(g["unrounded_profile_corners"] for g in geo.values()),
                max_sleeve_overlap_mm3=max(g["sleeve_overlap_mm3"] for g in geo.values()), invalid_solids=sum(g["invalid_solids"] for g in geo.values()),
                min_loadpath_mm2=min(paths) if paths else None, min_bed_margin_mm=min(g["min_bed_margin_mm"] for g in geo.values()),
                max_new_overlap_mm3=max((v["max_new_overlap_mm3"] for v in vals), default=0.0), min_distance_mm=min((v["min_distance_mm"] for v in vals), default=None),
                pairs_with_new_overlap=sum(1 for v in vals if v["max_new_overlap_mm3"] > 0), poses=rep["checks"]["clearance"]["notes"].split(";")[0],
                parts_without_step=rep["checks"]["geometry"]["bad_events"].get("parts_without_native_step", 0))


def check_cards(study, rep, result_id, source_refs):
    """Fit (clearance) and CAD-rule (geometry) evidence cards from the shared check's reports, with the command that recreates them; the totals too."""
    n = check_numbers(rep)
    cmd = check_command(study)
    fit = dict(id=f"{result_id}-fit", title="Pose-swept clearance against neighbouring parts (shared check: clearance)", kind="analysis",
               observation=(f"{n['poses']}: {n['pairs_with_new_overlap']} scoped-part/neighbour pairs with a new overlap over the pose set (largest {n['max_new_overlap_mm3']:.1f} mm3), "
                            f"smallest surface distance {n['min_distance_mm']} mm (capped at the search distance). Counts and distances, no verdict. Recreate: {cmd}"),
               measurements=[dict(name="Scoped parts", value=n["parts"], unit="parts"), dict(name="Pairs with a new overlap over the poses", value=n["pairs_with_new_overlap"], unit="pairs"),
                             dict(name="Largest new overlap vs standing", value=n["max_new_overlap_mm3"], unit="mm^3"), dict(name="Smallest surface distance", value=n["min_distance_mm"], unit="mm")],
               conditions=dict(poses=n["poses"], command=cmd), limitations="Rigid-body mesh sweep over the pose set; screw / driver rays are the separate screws and driver_access checks, not run here.", source_refs=source_refs)
    cad = dict(id=f"{result_id}-cad-rules", title="B-rep hygiene, sleeves, print volume and load-path slabs (shared check: geometry)", kind="analysis",
               observation=(f"{n['parts']} STEP parts: {n['invalid_solids']} invalid solids, {n['sliver_faces']} sliver faces, {n['micro_edges']} micro-edges, {n['acute_wedge_edges']} acute wedges, "
                            f"{n['unrounded_corners']} unrounded profile corners, {n['sleeves']} sleeves counted in the STEP, largest part-vs-sleeve overlap {n['max_sleeve_overlap_mm3']} mm3, "
                            f"largest STEP-vs-mesh volume difference {n['max_step_vs_mesh_pct']} %, smallest bed margin {n['min_bed_margin_mm']} mm. Recreate: {cmd}"),
               measurements=[dict(name="Invalid solids", value=n["invalid_solids"], unit="solids"), dict(name="Sliver faces", value=n["sliver_faces"], unit="faces"),
                             dict(name="Micro-edges", value=n["micro_edges"], unit="edges"), dict(name="Acute wedge edges", value=n["acute_wedge_edges"], unit="edges"),
                             dict(name="Sleeves counted", value=n["sleeves"], unit="sleeves"), dict(name="Largest part-vs-sleeve overlap", value=n["max_sleeve_overlap_mm3"], unit="mm^3"),
                             dict(name="Largest STEP-vs-mesh volume difference", value=n["max_step_vs_mesh_pct"], unit="%"),
                             *([dict(name="Thinnest load-path slab", value=n["min_loadpath_mm2"], unit="mm^2")] if n["min_loadpath_mm2"] is not None else [])],
               conditions=dict(command=cmd), limitations="Exact B-rep of the native STEP before slicing; parts without a native STEP are not audited.", source_refs=source_refs)
    return fit, cad, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study", required=True, choices=sorted(STUDIES))
    ap.add_argument("--report", help="report.json of run.py --checks fem[,fem_setup] on the study's parts")
    ap.add_argument("--out", help="default: artifacts/robot/body/<folder>/fem-evidence.json")
    ap.add_argument("--command", action="store_true", help="print the run.py command that produces the report")
    ap.add_argument("--check-command", action="store_true", help="print the run.py command that produces the geometry + clearance report of the record")
    a = ap.parse_args()
    if a.check_command:
        print(check_command(a.study)); return
    if a.command or not a.report:
        print(command(a.study)); return
    out = Path(a.out) if a.out else ROOT / "artifacts/robot/body" / STUDIES[a.study]["folder"] / "fem-evidence.json"
    ev = build(a.study, a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=1))
    print(f"{out}: {len(ev['parts'])} parts with results, {len(ev['parts_without_result'])} scoped parts without a result (no setup, or mesh-only part)")


if __name__ == "__main__":
    sys.exit(main())
