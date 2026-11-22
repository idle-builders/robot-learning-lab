"""Pair-rule scenarios (T-g72tnq, T-xfkwv2): the pair checks on the selected body, one hip-session finding each.

pairs-clearance-limbs   clearance on the parts with the worst whole-body cross-limb items (R palm lower, L sensor pocket pad,
                        R forearm back cover): pairs on two different limbs measured / left out, the worst cases left.
pairs-sweep-limbs-d15   collision_sweep on the two R thigh shells: contacts with a part on another limb (R shin vs L thigh shell,
                        forearm cover vs thigh shell were reported as cross-link overlaps), and the hip range corners of D-15
                        (thigh shells vs Honeycomb_Cell_08, about 800 mm3) as design overlaps or controller-limit contacts.
pairs-fit-class-links   fit_class on the thigh connector and stage 5: classes judged on two-link (running) pairs, seeded ones among them.
pairs-assembly-reach    assembly on the thigh connector: how far from the screw head the parts that close a driver path in lie.
pairs-fit-progress      fit on the hip stage-5 arm at --poses fit with a short pair budget: progress lines per pose, pairs stopped;
                        then the same run killed by memguard after 60 s: the report reads unfinished.

The part names here are the harness's choice of where the hip session saw each finding; the checks themselves know no parts."""
from __future__ import annotations

import json
import re
import sys

from evaluate import ROOT, basics, log_tail, run_tool, scenario, selected_manifest

THIGH = "R_thigh_roll_flip"
STAGE5 = "R_hip_pitch_to_roll_roll_flip"
SHELLS = "R_thigh_shell_front,R_thigh_shell_back"


def _limbs():
    """{part: limb root joint or None} of the selected manifest (bodylib.limb_of: the joint tree)."""
    sys.path.insert(0, str(ROOT / "robot/body"))
    from checks import core
    body = core.load_body(selected_manifest())
    lm = body.limb_of()
    return {n: lm.get(p["owning_link"]) for n, p in body.parts.items()}


def _two_limbs(limb, a, b):
    return bool(limb.get(a) and limb.get(b) and limb[a] != limb[b])


def _check(r, name):
    rep = r["report"] if isinstance(r["report"], dict) else {}
    return rep.get("checks", {}).get(name, {})


@scenario("pairs-clearance-limbs", "truth", "clearance: pairs on two different limbs (WBC self-collision limits) left out and counted, as the sweeps do")
def s_clear_limbs(sid):
    limb = _limbs()
    r = run_tool(sid, ["--parts", "R_palm_lower,L_sensor_pocket_pad_v19,R_fore_back_hard_cover", "--checks", "clearance"])
    c = _check(r, "clearance")
    pairs = [(a, m, w) for a, ms in (c.get("per_part") or {}).items() for m, w in ms.items()]
    cross = [(a, m, w) for a, m, w in pairs if _two_limbs(limb, a, m)]
    return dict(basics(r), bad_events=c.get("bad_events"),
                cross_limb_pairs_in_results=len(cross),
                cross_limb_max_new_overlap_mm3=max((w["max_new_overlap_mm3"] for _, _, w in cross), default=0.0),
                worst_cross_limb=sorted(([f"{a} x {m}", w["max_new_overlap_mm3"], w["pose_of_max"]] for a, m, w in cross), key=lambda x: -x[1])[:5],
                other_limb_pairs_listed=len(c.get("other_limb_pairs") or []),
                worst_cases=[[w["part"], w["mating_part"], w["value"], w["unit"], w["pose"]] for w in (c.get("worst_cases") or [])[:5]])


@scenario("pairs-sweep-limbs-d15", "truth", "collision_sweep on the R thigh shells: other-limb contacts, D-15 hip range corners as design overlaps or controller limits")
def s_sweep(sid):
    limb = _limbs()
    r = run_tool(sid, ["--parts", SHELLS, "--checks", "collision_sweep"], limit_gb=6, timeout_s=1800)
    c = _check(r, "collision_sweep")
    contacts = [x for pj in (c.get("per_joint") or {}).values() for x in pj["contacts"] if x["max_overlap_mm3"] > 0.05]
    cross = {f"{x['a']} x {x['b']}": round(x["max_overlap_mm3"], 1) for x in contacts if _two_limbs(limb, x["a"], x["b"])}
    hc = [w for w in (c.get("worst_cases") or []) if w["part"].startswith("Honeycomb_Cell_08") or w["mating_part"].startswith("Honeycomb_Cell_08")]
    # range_sweep with yaw and roll held at their range ends: the pitch end +45 is a D-15 corner
    h = run_tool(sid + "-hold", ["--parts", "R_thigh_shell_back", "--checks", "range_sweep", "--joints", "R_hip_pitch",
                                 "--hold", "R_hip_yaw=30", "--hold", "R_hip_roll=25"], limit_gb=6, timeout_s=1200)
    pj = (_check(h, "range_sweep").get("per_joint") or {}).get("R_hip_pitch") or {}
    held = dict(basics(h), first_contact_deg=pj.get("first_contact_deg"), contact_free_deg=pj.get("contact_free_deg"),
                corner_rows=[c for c in pj.get("curve", []) if c.get("controller_limit")],
                overlap_rows_last=[[c["deg"], c["overlap_mm3"]] for c in pj.get("curve", []) if c["overlap_mm3"] > 0][-4:])
    return dict(basics(r), bad_events=c.get("bad_events"), range_sweep_held_corner=held,
                cross_limb_contacts=cross,
                other_limb_pairs_left_out={jn: pj.get("other_limb_pairs_left_out") for jn, pj in (c.get("per_joint") or {}).items()},
                honeycomb_cell_08_design_cases=[[w["part"], w["mating_part"], round(w["value"], 1), w["unit"], w["pose"]] for w in hc],
                controller_limit_poses={jn: pj.get("controller_limit_poses") for jn, pj in (c.get("per_joint") or {}).items()},
                controller_limit_contacts=dict(list((c.get("controller_limit_contacts") or {}).items())[:8]),
                worst_cases=[[w["part"], w["mating_part"], round(w["value"], 1), w["unit"], w["pose"]] for w in (c.get("worst_cases") or [])[:6]])


@scenario("pairs-fit-class-links", "truth", "fit_class on the thigh connector and its stage-5 neighbour: no seeded class judged on a pair on two links (running clearance)")
def s_fit_class(sid):
    r = run_tool(sid, ["--parts", f"{THIGH},{STAGE5}", "--checks", "fit_class"])
    c = _check(r, "fit_class")
    per = c.get("per_part") or {}
    recs = [(a, m, x) for a, ms in per.items() for m, x in ms.items()]
    judged_running = [[a, m, x.get("fit_class"), x.get("class_source"), x["clearance_mm"]] for a, m, x in recs if not x["same_link"] and x.get("fit_class")]
    pair = (per.get(THIGH) or {}).get(STAGE5) or (per.get(STAGE5) or {}).get(THIGH)
    return dict(basics(r), bad_events=c.get("bad_events"), classes_judged_on_two_link_pairs=judged_running,
                thigh_x_stage5=pair, worst_cases=[[w["part"], w["mating_part"], w["value"], w["what"][:120]] for w in (c.get("worst_cases") or [])[:5]])


@scenario("pairs-assembly-reach", "truth", "assembly on the thigh connector: distance from the screw head of the parts that close a driver path in")
def s_assembly(sid):
    r = run_tool(sid, ["--parts", THIGH, "--checks", "assembly"])
    c = _check(r, "assembly")
    rows = []
    for asm in (c.get("assemblies") or {}).values():
        for st in asm.get("steps", []):
            for k, sc in (st.get("screws") or {}).items():
                for p in sc.get("closed_in_by", []):
                    rows.append([k, p, (sc.get("closed_in_from_mm") or {}).get(p), sc.get("path_length_mm")])
                for p, d in (sc.get("closed_in_beyond_reach") or {}).items():
                    rows.append([k, p + " (beyond reach)", d, sc.get("path_length_mm")])
    known = [x for x in rows if x[2] is not None and "beyond" not in x[1]]
    return dict(basics(r), bad_events=c.get("bad_events"), closing_in_entries=len(rows),
                closing_in_max_from_mm=max((x[2] for x in known), default=None),
                closing_in=sorted(rows, key=lambda x: -(x[2] or 0))[:12])


@scenario("pairs-fit-progress", "robust", "fit at --poses fit: progress per pose, a pair over --fit-pair-budget-s stops; a run killed after 60 s reads unfinished")
def s_fit(sid):
    a = run_tool(sid, ["--parts", STAGE5, "--checks", "fit", "--poses", "fit", "--fit-pair-budget-s", "8"], timeout_s=1200)
    ca = _check(a, "fit")
    lines = [ln for ln in a["log"].splitlines() if re.match(r"fit: pose \d+/\d+", ln.strip())]
    k = run_tool(sid + "-killed", ["--parts", STAGE5, "--checks", "fit", "--poses", "fit"], timeout_s=60)
    rk = k["report"] if isinstance(k["report"], dict) else {}
    return dict(budget_run=dict(basics(a), progress_lines=len(lines), first_last=[lines[0].strip(), lines[-1].strip()] if lines else None,
                                pairs_timed_out=ca.get("pairs_timed_out"), bad_events=ca.get("bad_events"),
                                budget_lines=[ln.strip() for ln in a["log"].splitlines() if "budget after" in ln][:5]),
                killed_run=dict(rc=k["rc"], exec_s=k["exec_s"], complete=rk.get("complete"), incomplete_running=rk.get("incomplete_running"),
                                stopped_in=str((rk.get("checks", {}).get("fit") or {}).get("stopped_in") or rk.get("stopped_in"))[:300],
                                progress_lines=len([ln for ln in k["log"].splitlines() if re.match(r"fit: pose \d+/\d+", ln.strip())]),
                                tail=log_tail(k, 4)))
