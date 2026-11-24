"""Compile the motor-refit evidence into one report: masses, interface gaps, FEM and collision sweeps against r1.

  .venv/bin/python -B robot/body/motor-refit/report.py
Reads the stored outputs of the retired per-study sweeps / FEM scripts (the current checks are robot/body/checks/run.py)
in the restored remote outputs in .cache/work/fdm-body-motor-refit/remote-out*/ and writes
artifacts/robot/body/motor-refit/report.json. Continuous values only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "robot/body"))
from checks import bodylib as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("motor-refit")   # configured values: robot/body/motor-refit/motor_refit.bom.yaml config:

R1 = ROOT / "artifacts/robot/body/assembly/assembly-parts.json"
OUT = ROOT / "artifacts/robot/body/motor-refit"
VERIFY = ROOT / ".cache/work/fdm-body-motor-refit/verify"
REMOTE = [VERIFY, ROOT / ".cache/work/fdm-body-motor-refit/remote-out", ROOT / ".cache/work/fdm-body-motor-refit/remote-out-second"]
LEG = CFG["LEG"]


def limb(link):
    if link in ("torso",):
        return "torso"
    if link in ("neck_yaw", "head"):
        return "head"
    side = link[:2]
    return side + ("leg" if link[2:] in LEG else "arm")


def servo_key(name):
    import re
    return re.sub(r"_(case|main_horn|rear_idler|upper_horn_reference|passive_idler_reference|servo_case_reference|output)$", "", name.replace("_servo_case_reference", "_case")).replace("upper_horn_reference", "").rstrip("_")


def classify(a, b, la, lb):
    ka, kb = servo_key(a), servo_key(b)
    if ka == kb or ka.replace("_case", "") == kb.replace("_case", "") or ("ST3025" in a and "ST3025" in b and a.rsplit("_ST3025", 1)[0] == b.rsplit("_ST3025", 1)[0]):
        return "servo model (case vs its own output)"
    if limb(la) != limb(lb):
        return "self-collision limit (different limbs or limb vs body)"
    return "design contact"


def verify_sweeps():
    """All joints swept locally on the final body, each contact classified."""
    out, limits = {}, []
    for f in sorted(VERIFY.glob("sweep-*.json")):
        for j, d in json.load(open(f))["joints"].items():
            rows = []
            for x in d["contacts"]:
                if x["max_overlap_mm3"] < 0.5:
                    continue
                cls = classify(x["a"], x["b"], x["link_a"], x["link_b"])
                row = dict(a=x["a"], b=x["b"], overlap_mm3=round(x["max_overlap_mm3"], 1), worst_pose=x["worst_pose"],
                           poses_overlapping=x["poses_overlapping"], measured="neutral pose, same link" if x["same_link"] else "over joint range", kind=cls)
                rows.append(row)
                if cls.startswith("self-collision"):
                    limits.append(dict(joint_swept=j, **row))
            out[j] = dict(poses=d["poses"], adjacent=d["adjacent"], pairs=rows)
    return out, limits
IGNORE = CFG["IGNORE"]
INTERFACES = [("R_thigh_combined", "R_knee_ST3025_case"), ("R_shin_combined", "R_knee_ST3025_output"),
              ("R_shin_combined", "R_ankle_pitch_ST3025_case"), ("R_ankle_coupler_combined", "R_ankle_pitch_ST3025_output"),
              ("R_hip_pitch_to_roll_combined", "R_hip_pitch_ST3025_output"), ("R_hip_yaw_to_pitch_combined", "R_hip_pitch_ST3025_case")]


def find(name):
    for d in REMOTE:
        if (d / name).exists():
            return d / name
    return None


def masses(man, kinds=("printed",)):
    return sum((p.get("mass_kg") or 0) for p in man["parts"] if p.get("role", "").startswith(kinds))


def sweep_table(tag):
    out = {}
    for name in ("leg", "hip", "left-hip"):
        a, b = find(f"sweep-refit-{name}.json"), find(f"sweep-r1-{name}.json")
        if not a or not b:
            continue
        ra, rb = json.load(open(a)), json.load(open(b))
        for j in ra["joints"]:
            key = lambda x: tuple(sorted((x["a"], x["b"])))
            A = {key(x): x for x in ra["joints"][j]["contacts"] if not any(t in x["a"] + x["b"] for t in IGNORE)}
            Bm = {key(x): x for x in rb["joints"][j]["contacts"] if not any(t in x["a"] + x["b"] for t in IGNORE)}
            rows = []
            for k in set(A) | set(Bm):
                va, vb = A.get(k, {}).get("max_overlap_mm3", 0.0), Bm.get(k, {}).get("max_overlap_mm3", 0.0)
                if max(va, vb) >= 0.5:
                    src = A.get(k) or Bm.get(k)
                    rows.append(dict(a=k[0], b=k[1], refit_mm3=round(va, 1), r1_mm3=round(vb, 1), measured="neutral pose, same link" if src["same_link"] else "over joint range",
                                     refit_worst_pose=A.get(k, {}).get("worst_pose"), refit_min_clearance_mm=A.get(k, {}).get("min_clearance_mm")))
            out[j] = dict(poses=ra["joints"][j]["poses"], adjacent=ra["joints"][j]["adjacent"],
                          pairs=sorted(rows, key=lambda r: -max(r["refit_mm3"], r["r1_mm3"])))
    return out


def fem_table():
    out = {}
    for key in ("shin", "ankle_coupler", "thigh", "hip_pitch_to_roll", "hip_yaw_to_pitch", "pelvis_cradle"):
        f = find(f"fem-{key}.json")
        if not f:
            continue
        d = json.load(open(f))["parts"][key]
        row = {}
        for which in ("r1", "refit"):
            m = d[which]["materials"]
            body = {c: v for c, v in m["PAHT_CF_dry"].items() if c != "servo_stall"}
            worst = min(body.items(), key=lambda kv: kv[1]["p99_strength_ratio"])
            row[which] = dict(stall_Nmm=d[which]["stall_Nmm"], body_worst_case=worst[0],
                              body_p99_strength_ratio_PAHT_CF_dry=worst[1]["p99_strength_ratio"],
                              body_p99_vm_MPa=worst[1]["p99_vm_MPa"], body_max_displacement_mm=worst[1]["max_displacement_mm"],
                              stall_p99_strength_ratio_PAHT_CF_dry=m["PAHT_CF_dry"].get("servo_stall", {}).get("p99_strength_ratio"),
                              stall_p99_vm_MPa=m["PAHT_CF_dry"].get("servo_stall", {}).get("p99_vm_MPa"),
                              min_p99_strength_ratio_PETG_CF=min(v["p99_strength_ratio"] for v in m["PETG_CF"].values()),
                              min_p99_layer_ratio_PAHT_CF_dry=min(v["p99_layer_ratio"] for v in m["PAHT_CF_dry"].values()),
                              mesh=d[which]["mesh"])
        out[key] = row
    return out


def gaps(man):
    B = F.Body(man); out = []
    for a, b in INTERFACES:
        A = F.Prepped(B.mesh(a)); m = B.mesh(b)
        d = A.tree.query(F.samples(m, 0.3, 60000), k=1)[0]
        ov = (F.to_mf(B.mesh(a)) ^ F.to_mf(m)).volume() if F.to_mf(B.mesh(a)) and F.to_mf(m) else None
        out.append(dict(part=a, servo_part=b, min_gap_mm=round(float(d.min()), 3), overlap_mm3=None if ov is None else round(ov, 2)))
    return out


def main():
    man, r1 = json.load(open(OUT / "assembly-parts.json")), json.load(open(F.src(R1)))
    P1 = {p["name"]: p for p in r1["parts"]}; P2 = {p["name"]: p for p in man["parts"]}
    edited = [n for n, p in P2.items() if p.get("native_step", "").startswith("artifacts/robot/body/motor-refit/")]
    changes = [dict(part=n, r1_mass_g=round((P1.get(n, {}).get("mass_kg") or 0) * 1000, 2), refit_mass_g=round((P2[n].get("mass_kg") or 0) * 1000, 2))
               for n in sorted(edited) if P2[n].get("role", "").startswith("printed")]
    remote = {}
    for tag in ("first", "second"):
        f = ROOT / f".bookkeeping/T-jrnxfn/remote/{tag}/submission.json"
        c = find("completion.json") if tag == "first" else None
        if f.exists():
            rec = json.load(open(f))
            remote[tag] = dict(run_id=rec["run_id"], job_id=rec["job_id"], provider_status=rec.get("provider_status"))
    comp = [json.load(open(d / "completion.json")) for d in REMOTE if (d / "completion.json").exists()]
    report = dict(
        schema="cove50.fdm-body-motor-refit.report.v1", idle_task="T-jrnxfn", source_r1_commit="48c38ceda",
        motor_map="robot/model/motor-map/motor-map.json",
        decisions={"M-44khpr": "locked three-tier motor set", "M-jrrxya": "owner: small hip shift + pelvis battery rework, keep the V body",
                   "M-jxjse2": "hip pitch output face +4 mm per side; lower-pelvis cells horizontal on the midline", "M-revtdr": "neck stays XM335",
                   "M-6r9xzc": "FEM and sweeps on the remote worker"},
        kinematics=dict(hip_pitch_spacing_mm=[118.0, 126.0], shifted_joints=["hip_roll", "knee", "ankle_pitch", "ankle_roll"],
                        shift_mm_per_side=4.0, knee_and_ankle_output_span_mm=[37.2, 51.0]),
        mass=dict(printed_r1_g=round(masses(r1) * 1000, 1), printed_refit_g=round(masses(man) * 1000, 1),
                  servos_20_leg_arm_joints_r1_g=20 * 55.0,  # r1 CAD: C047 (55 g) on every leg and arm joint; its manifest leaves most masses blank
                  servos_20_leg_arm_joints_refit_g=round(sum((p.get("mass_kg") or 0) for p in man["parts"] if p.get("motor") and "XM335" not in p["name"]) * 1000, 1),
                  servo_basis="M-44khpr catalogue masses: ST3025 124 g (case 120 + wheels 4), STS3250 74.5 g, C047 55 g; neck XM335 unchanged"),
        printed_part_changes=changes, interface_gaps=gaps(man), fem=fem_table(), collision_sweeps=sweep_table("refit"),
        fixes=dict(static=man.get("static_fixes", []), range=man.get("range_fixes", [])),
        static_audit=(lambda f: dict(parts=json.load(open(f))["parts"],
                                     pairs=[r for r in json.load(open(f))["pairs"] if (r["overlap_mm3"] or 0) >= 0.5 or r["overlap_mm3"] is None])
                      if f.exists() else None)(VERIFY / "static-audit.json"),
        final_sweeps=verify_sweeps()[0],
        worker_wall_s=[c.get("wall_s") for c in comp], remote_jobs=remote,
        r1_defects_found=[
            "L_hip_pitch_to_roll_combined overlaps L_roll_servo_case_reference by 324.4 mm3 in the r1 STEP (left roll servo flipped front-to-back, flange a plain mirror of the right)",
            "r1 hip roll servo case enters lower-pelvis cell 09 by 1,578 mm3 (cell 08 by 1,445 mm3 on the left) at hip yaw 30 / pitch 45; removed by the refit",
            "r1 whole-body verification covered 8 poses on non-watertight meshes, so it reported 0 mm3 for both"],
        open_items=[
            "Neck stays XM335 (owner, M-revtdr); the neck collar lost its front (6,033 -> 2,558 mm3) because the head skin sweeps through it over neck pitch: a visible change at the front of the neck.",
            "Self-collision limits (owner decision): corner contacts between limbs, or a limb and the body/head, are listed in self-collision-limits.json for the WBC; per-joint ranges are unchanged.",
            "Servo reference models: each case touches its own horn/idler/output by 1-5 mm3 (model detail of the supplied servo references, not a printed-part contact).",
            "Hip pitch-to-roll flange at STS3250 hip-roll stall stays below its r1 margin at C047 stall (see FEM); the ankle coupler now exceeds r1.",
            "New M2/M3 holes carry no pressed steel sleeves; 18 r1 sleeves dropped with the old C047 screw positions.",
            "Front shell lower attachment: two M2 posts at z 257.5 with screw heads on the lower belly.",
            "Mesh-only notches (forearm covers, neck collar, head skin, shin front covers): the mesh is the part; the head skin and shin front cover STEPs predate their notch.",
            "Supplier meshes that are not closed (e.g. the XM335 neck case, compute board) could not be measured by the static audit.",
            "The WBC training plant uses R3 kinematics: the 4 mm hip shift per side needs a model re-export before the WBC uses this body."],
    )
    sweeps, limits = verify_sweeps()
    json.dump(dict(schema="cove50.self-collision-limits.v1", source="T-jrnxfn local sweeps", decision="owner 2026-11-12: corner contacts handled as WBC self-collision limits",
                   pairs=limits), open(OUT / "self-collision-limits.json", "w"), indent=1)
    from collections import Counter
    report["final_sweep_summary"] = {j: dict(Counter(r["kind"] for r in d["pairs"])) for j, d in sweeps.items()}
    report["design_contacts_remaining"] = [dict(joint=j, **r) for j, d in sweeps.items() for r in d["pairs"] if r["kind"] == "design contact"]
    json.dump(report, open(OUT / "report.json", "w"), indent=1)
    print(json.dumps(dict(mass=report["mass"], gaps=report["interface_gaps"], joints={j: len(v["pairs"]) for j, v in report["collision_sweeps"].items()}), indent=1))


if __name__ == "__main__":
    main()
