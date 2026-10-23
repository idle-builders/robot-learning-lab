"""Build the inspector report from saved Cove transfer evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-transfer-20261008"


def read(name: str) -> dict | None:
    path = OUT / name
    return json.loads(path.read_text()) if path.is_file() else None


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def worlds(report: dict | None) -> dict | None:
    if not report:
        return None
    rows = report["worlds"]
    mean = lambda key: round(sum(w[key] for w in rows) / len(rows), 4)
    return {
        "worlds": len(rows),
        "survived_8s": sum(w["ended_by_timeout"] for w in rows),
        "one_meter_alternating": report["one_meter_with_alternating_steps"],
        "mean_forward_m": mean("forward_m"),
        "mean_lateral_m": mean("lateral_m"),
        "mean_yaw_rad": mean("net_yaw_rad"),
        "max_body_contact_fraction": max(w["fraction_frames_with_body_contact"] for w in rows),
    }


def main() -> None:
    ledger = read("body-ledger.json")
    catalog = read("policy-inventory.json")
    old = read("baseline-old-seed144.json")
    direct = read("baseline-direct-seed144.json")
    matrix = read("varied150-matrix.json")
    stop = read("varied150-walk-stop.json")
    mapped_stand = read("stand-mapped-baseline-seed144.json")
    passive_stand = read("stand-zero-seed144.json")
    if ledger is None or catalog is None:
        raise ValueError("Missing selected body ledger or retained policy inventory")
    forward = OUT / "cove50-forward-varied150.pt"
    report = {
        "schema": "robot-learning-lab.cove50-policy-transfer-report.v1",
        "status": "in_progress",
        "selected_design": {
            "id": ledger["selected_design_id"],
            "pointer_sha256": ledger["selected_design_sha256"],
            "cad_generator_sha256": ledger["cad_generator_sha256"],
            "report_sha256": ledger["source_report_sha256"],
        },
        "body": {
            "visual_xml": ledger["generated_body"],
            "visual_xml_sha256": ledger["generated_body_sha256"],
            "training_xml": ledger["training_body"],
            "training_xml_sha256": ledger["training_body_sha256"],
            "compiled_mass_kg": ledger["compiled_mass_kg"],
            "source_mass_kg": ledger["source_body_mass_kg"],
            "mass_model": ledger["mass_model"],
            "contact_model": ledger["contact_model"],
            "qualification": "provisional simulation: unmeasured inertia, servo impedance, thermal duty and full-frame swept clearance",
        },
        "retained_catalog": {
            "policy_count": catalog["policy_count"],
            "registry_sha256": catalog["registry_sha256"],
            "status": "historical policies need individual Cove rollout gates",
        },
        "paired_checkpoint_baseline": {
            "old_body": worlds(old),
            "direct_body": worlds(direct),
            "same_initial_state": old["initial_state_sha256"] == direct["initial_state_sha256"] if old and direct else None,
            "same_first_action": old["first_action_sha256"] == direct["first_action_sha256"] if old and direct else None,
        },
        "forward_continuation": {
            "checkpoint": str(forward.relative_to(ROOT)),
            "checkpoint_sha256": sha(forward),
            "updates": 150,
            "worlds": 512,
            "environment_steps": 1843200,
            "training_variance": {
                "mass_and_inertia_scale": [0.9, 1.1],
                "sole_sliding_friction": [0.65, 1.15],
                "effort_scale": [0.9, 1.05],
                "pd_gain_scale": [0.9, 1.1],
                "command_lag_physics_steps": [0, 4],
                "forward_m_s": [0.2, 0.45],
                "yaw_rad_s": [-0.35, 0.35],
                "standing_fraction": 0.10,
                "small_root_and_joint_reset_noise": True,
                "status": "sensitivity ranges, not measured hardware distributions",
            },
            "fresh_matrix": ([{
                "case": c["case"], "survived_8s": c.get("survived_8s"),
                "mean_forward_m": c.get("mean_forward_m"),
                "mean_lateral_m": c.get("mean_lateral_m"),
                "mean_yaw_rad": c.get("mean_yaw_rad"),
                "max_body_contact_fraction": c.get("max_body_contact_fraction"),
                "report": c["report"],
            } for c in matrix["cases"]] if matrix else None),
            "walk_stop_max_travel_m": (max(w["stop"]["travel_after_command_m"] for w in stop["worlds"])
                                       if stop else None),
            "motor_case_screen": "varied150-forward-seed144-cad-cases.json",
            "saturation_note": "This first matrix used a fixed 1.56 Nm gate, so its 10 V zero saturation figures are invalid; the evaluator was corrected for subsequent matrices.",
        },
        "planar_continuation": "see planar200v3-planar-matrix.json",
        "standing": {
            "mapped_old_policy": mapped_stand["stand_after"] if mapped_stand else None,
            "passive_zero_action": passive_stand["stand_after"] if passive_stand else None,
            "new_checkpoint": "stand-loweffort198-seed144.json; inferior to passive zero action",
        },
        "unresolved": [
            "Combined low-grip, altered mass, 10 V and 10 ms cases have large path drift.",
            "Forward checkpoint barely turns in place.",
            "Leg duty near catalog-rated effort remains unqualified physically.",
            "Posture, arm, foot, manipulation and composition skills have not passed integrated Cove rollout gates.",
        ],
    }
    outcomes = read("transfer-outcomes.json")
    if outcomes:
        report["status"] = "completed_with_rejected_integrated_transfer"
        report["qualified_skill_count"] = outcomes["qualified_count"]
        report["policy_outcomes"] = "artifacts/ape-cove50-transfer-20261008/transfer-outcomes.json"
        report["retained_catalog"]["status"] = "all 29 policies given an evidence-backed Cove disposition"
        report["fresh_matrices"] = {
            "nominal_and_wide_forward": "robust10v200-fixed-matrix.json",
            "backward": "backward-varied300-matrix.json",
            "planar": "planar200v3-planar-matrix.json",
            "squat": "squat-counterbalance298-posture-arm-matrix.json",
            "arms": "arm150-posture-arm-matrix.json",
            "geometry": "skill-geometry-audit.json",
        }
        report["decision"] = (
            "Selected direct-frame body remains a design candidate. Its learned skill "
            "portfolio did not pass robust simulated control, collision and actuator gates."
        )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
