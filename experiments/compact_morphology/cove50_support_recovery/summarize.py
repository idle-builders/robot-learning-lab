"""Assemble a compact evidence matrix from retained Cove support probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-support-recovery-20261009"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
POINTER = ROOT / "robot/ape-current-design.json"


def case(name: str, method: str, condition: str, *, kind: str) -> dict | None:
    path = OUT / name
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    row = {"method": method, "condition": condition,
           "report": str(path.relative_to(ROOT))}
    if kind == "static":
        margin = data["stance_sole_half_widths_xy_m"][1] - abs(data["com_in_stance_sole_xy_m"][1])
        row.update(outcome="Pose screened", duration_s=None,
                   single_support_s=None, rated_leg_fraction=None,
                   slip_mm=None, contact_frames=None, self_contact_frames=None,
                   com_margin_mm=round(1000 * margin, 2),
                   swing_clearance_mm=round(1000 * data["swing_sole_lowest_point_z_m"], 2))
    elif kind == "xml":
        row.update(outcome="Reset held" if data["result"] == "held" else "Reset fell",
                   duration_s=data["duration_s"],
                   single_support_s=data["duration_s"] if data["result"] == "held" else None,
                   rated_leg_fraction=data["rated_leg_fraction"],
                   slip_mm=round(1000 * data["max_stance_slip_m"], 1),
                   contact_frames=data["non_sole_floor_contact_frames"],
                   self_contact_frames=len(data["self_contact_pairs"]),
                   com_margin_mm=None, swing_clearance_mm=None)
    elif kind == "native":
        held = (data["duration_s"] >= 2.98 and data["longest_single_support_s"] >= 2.0
                and data.get("nonsole_ground_contact_frames", 1) == 0
                and data["max_hand_force_n"] < 1.0)
        row.update(outcome=("Hold with collision" if data["body_contact_frames"] else "Reset held")
                   if held else "Reset failed",
                   duration_s=data["duration_s"],
                   single_support_s=data["longest_single_support_s"],
                   rated_leg_fraction=data["rated_leg_fraction"],
                   slip_mm=round(1000 * data["max_stance_slip_m"], 1),
                   contact_frames=data.get("nonsole_ground_contact_frames"),
                   self_contact_frames=data["body_contact_frames"],
                   com_margin_mm=None, swing_clearance_mm=None)
    elif kind == "transition":
        row.update(outcome="Path complete" if data["final_two_foot_stand"] and
                   data["longest_single_support_s"] >= 2 else "Path failed",
                   duration_s=data["duration_s"],
                   single_support_s=data["longest_single_support_s"],
                   rated_leg_fraction=data["rated_leg_fraction"],
                   slip_mm=round(1000 * data["max_stance_slip_m"], 1),
                   contact_frames=data.get("nonsole_floor_contact_frames",
                                           data.get("body_contact_frames")),
                   self_contact_frames=None,
                   com_margin_mm=None, swing_clearance_mm=None)
    elif kind == "native_transition":
        slip_m = (data["max_stance_slip_m"] if "max_stance_slip_m" in data
                  else data["max_right_slip_m"])
        complete = (data["final_two_foot_stand"] and
                    data["longest_single_support_s"] >= 2.0 and
                    slip_m <= 0.015 and
                    data["rated_leg_fraction"] <= 0.10 and
                    data["nonsole_floor_contact_frames"] == 0 and
                    data["hand_contact_frames"] == 0 and
                    not data["cpu_audited_self_contact_pairs"])
        row.update(outcome="Path complete" if complete else "Path failed",
                   duration_s=data["duration_s"],
                   single_support_s=data["longest_single_support_s"],
                   rated_leg_fraction=data["rated_leg_fraction"],
                   slip_mm=round(1000 * slip_m, 1),
                   contact_frames=data["nonsole_floor_contact_frames"],
                   self_contact_frames=sum(pair["sampled_frames"] for pair in
                                           data["cpu_audited_self_contact_pairs"]),
                   com_margin_mm=None, swing_clearance_mm=None)
    return row


def main() -> None:
    featured = [
        "right-residual100-freeze.json", "right-freeze-stress3701.json",
        "left-domain125-freeze-nominal2905.json", "left-domain125-freeze-stress.json",
        "right-residual100-seed2144.json", "right-low-swing-xml-hold.json",
        "right-transition-phase199-seed2144.json",
        "right-transition-phase199-cleararm.json",
        "right-transition-phase199-cleararm-stress3701.json",
        "left-transition-domain125-seed2905.json",
        "left-transition-domain125-stress3701.json",
        "right-transition-urgent350-seed2144.json",
        "right-transition-frozen125-seed2144.json",
        "left-transition-urgent200-seed2905.json",
        "left-transition-urgent324-seed2905.json",
    ]
    specs = [
        ("right-low-swing-geometry.json", "Static geometry", "Right · low swing", "static"),
        ("left-low-swing-geometry.json", "Static geometry", "Left · low swing", "static"),
        ("right-low-swing-xml-hold.json", "XML capped servo", "Right · pose reset", "xml"),
        ("left-low-swing-xml-hold.json", "XML capped servo", "Left · pose reset", "xml"),
        ("right-low-swing-native-fall.json", "Native motor target", "Right · nominal · seed 2144", "native"),
        ("left-low-swing-native-seed2905.json", "Native motor target", "Left · nominal · seed 2905", "native"),
        ("right-low-swing-stress-seed3701.json", "Native motor target", "Right · heavy/grip/10 V/10 ms · seed 3701", "native"),
        ("right-low-swing-oldactor-seed2144.json", "Retained PPO actor", "Right · nominal · seed 2144", "native"),
        ("right-holdfirst-225-seed2144.json", "Continued PPO actor", "Right · high swing · seed 2144", "native"),
        ("right-residual25-seed2144.json", "Residual PPO actor", "Right · low swing · checkpoint 25", "native"),
        ("right-residual25-seed2905.json", "Residual PPO actor", "Right · low swing · checkpoint 25 · seed 2905", "native"),
        ("right-residual25-stress3701.json", "Residual PPO actor", "Right · heavy/grip/10 V/10 ms · checkpoint 25", "native"),
        ("right-residual50-seed2144.json", "Residual PPO actor", "Right · low swing · checkpoint 50", "native"),
        ("right-residual75-seed2144.json", "Residual PPO actor", "Right · low swing · checkpoint 75", "native"),
        ("right-residual100-seed2144.json", "Residual PPO actor", "Right · low swing · checkpoint 100", "native"),
        ("right-residual100-stress3701.json", "Residual PPO actor", "Right · heavy/grip/10 V/10 ms · checkpoint 100", "native"),
        ("right-residual100-nominal-jitter4909.json", "Residual PPO actor", "Right · nominal jitter · seed 4909", "native"),
        ("right-residual100-stress-jitter4909.json", "Residual PPO actor", "Right · stress jitter · seed 4909", "native"),
        ("right-residual100-stress-jitter5911.json", "Residual PPO actor", "Right · stress jitter · seed 5911", "native"),
        ("right-residual100-freeze.json", "Swing ablation", "Right · nominal · checkpoint 100", "native"),
        ("right-residual100-roll72.json", "Swing roll constraint", "Right · nominal · -0.72 rad", "native"),
        ("right-residual100-roll68.json", "Swing roll constraint", "Right · nominal · -0.68 rad", "native"),
        ("right-freeze-nominal-jitter4909.json", "Composed PPO hold", "Right · nominal jitter · seed 4909", "native"),
        ("right-freeze-stress3701.json", "Composed PPO hold", "Right · stress · seed 3701", "native"),
        ("right-freeze-stress-jitter4909.json", "Composed PPO hold", "Right · stress jitter · seed 4909", "native"),
        ("right-freeze-stress-jitter5911.json", "Composed PPO hold", "Right · stress jitter · seed 5911", "native"),
        ("right-residual125-seed2144.json", "Residual PPO actor", "Right · low swing · checkpoint 125", "native"),
        ("right-residual149-seed2144.json", "Residual PPO actor", "Right · low swing · checkpoint 149", "native"),
        ("left-residual99-nominal.json", "Residual PPO actor", "Left · nominal · checkpoint 99", "native"),
        ("left-residual99-stress.json", "Residual PPO actor", "Left · stress · checkpoint 99", "native"),
        ("left-freeze99-nominal.json", "Composed PPO hold", "Left · nominal · checkpoint 99", "native"),
        ("left-freeze99-stress.json", "Composed PPO hold", "Left · stress · checkpoint 99", "native"),
        ("left-domain125-freeze-nominal2905.json", "Composed PPO hold", "Left · nominal · domain checkpoint 125", "native"),
        ("left-domain125-freeze-nominal-jitter4909.json", "Composed PPO hold", "Left · nominal jitter · checkpoint 125", "native"),
        ("left-domain125-freeze-stress.json", "Composed PPO hold", "Left · stress · checkpoint 125", "native"),
        ("left-domain125-freeze-stress-jitter4909.json", "Composed PPO hold", "Left · stress jitter · seed 4909", "native"),
        ("left-domain125-freeze-stress-jitter5911.json", "Composed PPO hold", "Left · stress jitter · seed 5911", "native"),
        ("left-residual25-seed2905.json", "Residual PPO actor", "Left · low swing · checkpoint 25", "native"),
        ("left-residual25-stress3701.json", "Residual PPO actor", "Left · heavy/grip/10 V/10 ms · checkpoint 25", "native"),
        ("right-transition-all-2s.json", "XML target ramp", "Upright → right · 2 s", "transition"),
        ("right-transition-staged-4s.json", "XML staged ramp", "Upright → right · 4 s", "transition"),
        ("right-shift0.24-lift4.json", "XML shift + lift", "Upright → right · shifted · 4 s", "transition"),
        ("right-transition-phase125-seed2144.json", "Native transition PPO", "Upright → right · checkpoint 125", "native_transition"),
        ("right-transition-phase175-seed2144.json", "Native transition PPO", "Upright → right · checkpoint 175", "native_transition"),
        ("right-transition-phase199-seed2144.json", "Native transition PPO", "Upright → right · checkpoint 199", "native_transition"),
        ("right-transition-phase199-lift3.json", "Native transition PPO", "Upright → right · 3 s lift · checkpoint 199", "native_transition"),
        ("right-transition-phase199-lift4.json", "Native transition PPO", "Upright → right · 4 s lift · checkpoint 199", "native_transition"),
        ("right-transition-phase199-cleararm.json", "Clearance detour PPO", "Upright → right · 4 s lift · 0.25 rad arm detour", "native_transition"),
        ("right-transition-phase199-cleararm-stress3701.json", "Clearance detour PPO", "Upright → right · stress · arm detour · seed 3701", "native_transition"),
        ("right-transition-cleararm225-seed2144.json", "Clearance detour PPO", "Upright → right · phase reward · checkpoint 225", "native_transition"),
        ("right-transition-staged199-baseline.json", "Staged clearance PPO", "Upright → right · staged baseline · checkpoint 199", "native_transition"),
        ("right-transition-staged199-freeze.json", "Staged clearance PPO", "Upright → right · staged frozen swing · checkpoint 199", "native_transition"),
        ("right-transition-staged225-seed2144.json", "Staged clearance PPO", "Upright → right · staged reward · checkpoint 225", "native_transition"),
        ("right-transition-staged250-slow.json", "Staged clearance PPO", "Upright → right · 6 s lift · checkpoint 250", "native_transition"),
        ("right-transition-staged300-seed2144.json", "Staged clearance PPO", "Upright → right · staged reward · checkpoint 300", "native_transition"),
        ("right-transition-urgent350-seed2144.json", "Staged urgent PPO", "Upright → right · urgent reward · checkpoint 350", "native_transition"),
        ("right-transition-urgent400-seed2144.json", "Staged urgent PPO", "Upright → right · urgent reward · checkpoint 400", "native_transition"),
        ("right-transition-frozen125-seed2144.json", "Frozen swing PPO", "Upright → right · staged fixed swing · checkpoint 125", "native_transition"),
        ("right-transition-frozen125-slow6.json", "Frozen swing PPO", "Upright → right · staged fixed swing · 6 s lift", "native_transition"),
        ("right-transition-staged-frozen150-seed2144.json", "Frozen swing PPO", "Upright → right · staged fixed swing · checkpoint 150", "native_transition"),
        ("right-transition-staged-frozen199-seed2144.json", "Frozen swing PPO", "Upright → right · staged fixed swing · checkpoint 199", "native_transition"),
        ("left-transition-domain125-seed2905.json", "Native transition PPO", "Upright → left · nominal · domain checkpoint 125", "native_transition"),
        ("left-transition-domain125-stress3701.json", "Native transition PPO", "Upright → left · stress · domain checkpoint 125", "native_transition"),
        ("left-transition-urgent175-seed2905.json", "Staged urgent PPO", "Upright → left · urgent reward · checkpoint 175", "native_transition"),
        ("left-transition-urgent200-seed2905.json", "Staged urgent PPO", "Upright → left · urgent reward · checkpoint 200", "native_transition"),
        ("left-transition-urgent250-seed2905.json", "Staged urgent PPO", "Upright → left · urgent reward · checkpoint 250", "native_transition"),
        ("left-transition-urgent275-seed2905.json", "Staged urgent PPO", "Upright → left · urgent reward · checkpoint 275", "native_transition"),
        ("left-transition-urgent275-slow6.json", "Staged urgent PPO", "Upright → left · urgent reward · 6 s lift", "native_transition"),
        ("left-transition-urgent324-seed2905.json", "Staged urgent PPO", "Upright → left · urgent reward · checkpoint 324", "native_transition"),
        ("left-transition-stability350-seed2905.json", "Stability reward PPO", "Upright → left · COM/gravity/slip continuation · checkpoint 350", "native_transition"),
        ("left-transition-stability375-seed2905.json", "Stability reward PPO", "Upright → left · COM/gravity/slip continuation · checkpoint 375", "native_transition"),
        ("left-transition-stability423-seed2905.json", "Stability reward PPO", "Upright → left · COM/gravity/slip continuation · checkpoint 423", "native_transition"),
        ("right-transition-frozen150-seed2144.json", "Frozen swing PPO", "Upright → right · checkpoint 150", "native_transition"),
        ("right-transition-phase275-seed2144.json", "Native transition PPO", "Upright → right · checkpoint 275", "native_transition"),
        ("right-transition-phase348-seed2144.json", "Native transition PPO", "Upright → right · checkpoint 348", "native_transition"),
    ]
    cases = [c for name, method, condition, kind in specs
             if (c := case(name, method, condition, kind=kind)) is not None]
    for row in cases:
        row["featured"] = Path(row["report"]).name in featured
    cases.sort(key=lambda row: (0, featured.index(Path(row["report"]).name))
               if row["featured"] else (1, row["method"], row["condition"]))
    matrix = {
        "schema": "robot-learning-lab.cove50-support-matrix.v1",
        "selected_pointer_sha256": hashlib.sha256(POINTER.read_bytes()).hexdigest(),
        "selected_body_sha256": hashlib.sha256(BODY.read_bytes()).hexdigest(),
        "cases": cases,
    }
    (OUT / "evaluation-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    by_name = {Path(row["report"]).name: row for row in cases}
    passing_paths = [row for row in cases if row["outcome"] == "Path complete"
                     and row["method"] in ("Native transition PPO",
                                           "Clearance detour PPO", "Frozen swing PPO",
                                           "Staged clearance PPO", "Staged urgent PPO",
                                           "Stability reward PPO")]
    path_coverage = {
        f"{side}_{condition}": any(
            f"Upright → {side}" in row["condition"] and
            (("stress" in row["condition"]) == (condition == "stress"))
            for row in passing_paths)
        for side in ("right", "left") for condition in ("nominal", "stress")
    }
    gates = {
        "right_nominal_reset": by_name.get("right-residual100-freeze.json", {}).get("outcome") == "Reset held",
        "right_stress_reset": all(by_name.get(name, {}).get("outcome") == "Reset held" for name in
                                  ("right-freeze-stress3701.json", "right-freeze-stress-jitter4909.json",
                                   "right-freeze-stress-jitter5911.json")),
        "left_nominal_reset": by_name.get("left-domain125-freeze-nominal2905.json", {}).get("outcome") == "Reset held",
        "left_stress_reset": all(by_name.get(name, {}).get("outcome") == "Reset held" for name in
                                 ("left-domain125-freeze-stress.json", "left-domain125-freeze-stress-jitter4909.json",
                                  "left-domain125-freeze-stress-jitter5911.json")),
        "upright_to_hold_and_return": all(path_coverage.values()),
    }
    report = {
        "schema": "robot-learning-lab.cove50-support-recovery.v1",
        "selected_pointer_sha256": matrix["selected_pointer_sha256"],
        "selected_body_sha256": matrix["selected_body_sha256"],
        "scope": "Upright two-foot stand to commanded left/right single support >=2 s and back, without hand/body ground contact",
        "outcome": "full support task demonstrated" if all(gates.values()) else
                   "partial raised-foot holds; full support task unqualified",
        "gates": gates,
        "upright_path_coverage": path_coverage,
        "interpretation": [
            "The static endpoint screen and XML capped-servo reset holds show a mechanically plausible one-foot pose on both sides.",
            "XML position servos are a feasibility probe; native MJLab DC motor replays are the controller gate.",
            "Scripted XML target ramps and the first native transition PPO branch failed from an upright two-foot stand; no learned stand-to-lift-and-return controller is qualified.",
            "Mirrored native upright trials fail on both sides. The left held-out heavy/low-grip/10 V/10 ms trial spends 98.8% of frames at rated leg effort before falling; a right detour stress trial reaches 85.5% rated leg time and still lacks sustained exclusive support.",
            "Kinematic simultaneous and staged interpolations intersect the swing-side elbow and hip motor cases on both sides; mirrored 0.25 rad swing-shoulder-roll detours removed self-contact in all 101 samples of each schedule, but dynamic transfer remains unqualified.",
            "Staged phase shaping can survive 11 s while keeping both feet down, so episode length and training reward are not evidence of transfer. Fixing swing-leg residuals forces initial lift but the right replay loses balance around 3.7 s with about 95 mm stance slip and 69% rated leg time; a slower 6 s lift only delays the same failure.",
            "The later left phase checkpoint reduces rated leg time to 4.6% in its reserved nominal replay, yet achieves only 0.12 s exclusive support before a 112.8 mm stance slip and nonsole floor contact. The current left gate is timing and contact stability in the two-foot-to-one-foot transition; these data do not prove that the actuator envelope alone is insufficient.",
            "A further 100-update left continuation with stronger COM, gravity and slip rewards did not clear the transfer gate: final checkpoint 423 reaches only 0.16 s exclusive support, slips 140 mm and touches the ground with a hand. It is rejected despite a longer survival time.",
            "CPU MuJoCo contact attribution finds repeated thigh-torso self collision in the unconstrained nominal residual PPO hold; a duration pass with that collision is not a controller qualification.",
        ],
        "kinematic_transition_clearance_screens": [
            str((OUT / f"{side}-phase-clearance-screen.json").relative_to(ROOT))
            for side in ("right", "left")],
        "qualification_gate": {
            "native_single_support_s": 2.0,
            "no_hand_or_nonsole_ground_contact": True,
            "no_self_contact": True,
            "must_start_upright_and_return": True,
            "nominal_and_held_out": True,
            "stance_slip_m_at_most": 0.015,
            "rated_leg_fraction_at_most": 0.10,
        },
        "selected_raised_foot_controllers": [
            {
                "stance": side,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "action_contract": "PPO residual at 0.2 rad scale around gravity-preloaded joint targets; all five swing-leg residuals fixed to zero at replay",
                "pose": f"artifacts/ape-cove50-support-recovery-20261009/{side}-low-swing-endpoint.json",
                "preload": f"artifacts/ape-cove50-support-recovery-20261009/{side}-low-swing-preload.json",
                "training_params": f"artifacts/ape-cove50-support-recovery-20261009/{params}/",
                "training_variance": variation,
            }
            for side, checkpoint, params, variation in (
                ("right", OUT / "right-residual100.pt", "right-residual-params",
                 "256 worlds; 1 degree tilt and 0.005 rad joint reset noise; fixed nominal body/motor during training"),
                ("left", OUT / "left-domain125.pt", "left-domain-params",
                 "256 worlds; 1 degree tilt and 0.005 rad joint reset noise; pseudo-inertia alpha ln(0.85)/2 to ln(1.15)/2, absolute sole friction 0.55–1.15, effort scale 0.83–1.05, PD scale 0.9–1.1, command lag 0–10 ms"),
            )
        ],
        "limits": [
            "All results are in simulation on the selected provisional CAD-derived body and a catalog-bounded DC motor sensitivity model.",
            "The raised-foot replay does not establish a stand-to-lift or return controller.",
            "Contact-free means no modelled self-contact or non-sole floor/hand contact in sampled replay frames; it does not measure physical clearances.",
        ],
        "matrix": str((OUT / "evaluation-matrix.json").relative_to(ROOT)),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "outcome": report["outcome"]}))


if __name__ == "__main__":
    main()
