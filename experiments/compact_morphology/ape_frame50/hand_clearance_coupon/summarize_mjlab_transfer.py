"""Verify paired native MJLab squat replay on selected and candidate Cove hands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010"
TRIALS = OUT / "trials"
PLAN = OUT / "plan.json"
WINDOWS = ("initial_hold", "middle_hold", "final_hold")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_remote_hashes() -> None:
    manifest = TRIALS / "remote-sha256.txt"
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        if not path.exists() or sha(path) != expected:
            raise ValueError(f"Remote output missing or hash mismatch: {path}")


def pass_window(window: dict) -> bool:
    return (window["both_feet_grounded_fraction"] >= .95 and
            window["body_contact_fraction"] == 0 and
            window["hand_contact_frames"] == 0 and
            window["saturated_leg_fraction"] <= .10 and
            window["max_tilt_deg"] <= 5)


def main() -> None:
    plan = json.loads(PLAN.read_text())
    identity = {
        "selected_pointer_sha256": "robot/ape-current-design.json",
        "selected_body_sha256": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
        "candidate_body_sha256": "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml",
        "checkpoint_sha256": "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
        "evaluator_sha256": "experiments/compact_morphology/mjlab_ape/evaluate_posture.py",
        "motor_config_source_sha256": "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    }
    for key, relative in identity.items():
        if sha(ROOT / relative) != plan[key]:
            raise ValueError(f"Source identity changed: {key}")
    verify_remote_hashes()
    cases = {}
    rows = []
    for seed in (2906, 7911):
        for body in ("selected", "candidate"):
            case = f"{body}-{seed}"
            result = json.loads((TRIALS / f"{case}.json").read_text())
            expected = plan[f"{body}_body_sha256"]
            if result["body_sha256"] != expected:
                raise ValueError(f"Wrong body in {case}")
            if result["seed"] != seed or result["motor_rail_v"] != 12.0 or result["command_delay_ms"] != 0:
                raise ValueError(f"Changed command/seed in {case}")
            if (not result["start_squat"] or not result["settled_squat_reset"] or
                    not result["expose_squat_blend"] or not result["mask_action_history"] or
                    not result["squat_active_mask"] or result["mode"] != "cycle"):
                raise ValueError(f"Changed squat replay contract in {case}")
            cases[case] = result
            windows = {name: result[name] for name in WINDOWS}
            row = {
                "case": case, "body": body, "seed": seed,
                "duration_s": result["duration_s"],
                "ended_early": result["ended_early"],
                "all_hold_windows_pass": all(pass_window(w) for w in windows.values()),
                "min_both_feet_fraction": min(w["both_feet_grounded_fraction"] for w in windows.values()),
                "max_tilt_deg": max(w["max_tilt_deg"] for w in windows.values()),
                "max_saturated_leg_fraction": max(w["saturated_leg_fraction"] for w in windows.values()),
                "hand_contact_frames": sum(w["hand_contact_frames"] for w in windows.values()),
                "max_body_contact_fraction": max(w["body_contact_fraction"] for w in windows.values()),
                "hold_pelvis_z_m": {name: windows[name]["mean_pelvis_z_m"] for name in WINDOWS},
                "report": str((TRIALS / f"{case}.json").relative_to(ROOT)),
                "state": str((TRIALS / f"{case}.npz").relative_to(ROOT)),
                "video": (str((TRIALS / f"{case}.mp4").relative_to(ROOT))
                          if (TRIALS / f"{case}.mp4").exists() else None),
            }
            row["base_gate_pass"] = (
                row["duration_s"] >= 9.9 and not row["ended_early"] and
                row["all_hold_windows_pass"])
            rows.append(row)
    comparison = []
    for seed in (2906, 7911):
        selected = next(r for r in rows if r["case"] == f"selected-{seed}")
        candidate = next(r for r in rows if r["case"] == f"candidate-{seed}")
        deltas = {name: round(candidate["hold_pelvis_z_m"][name] -
                              selected["hold_pelvis_z_m"][name], 5)
                  for name in WINDOWS}
        comparison.append({
            "case": f"matched seed {seed}", "seed": seed,
            "selected_gate_pass": selected["base_gate_pass"],
            "candidate_base_gate_pass": candidate["base_gate_pass"],
            "candidate_pelvis_delta_m": deltas,
            "candidate_height_within_30mm": all(abs(value) <= .03 for value in deltas.values()),
            "candidate_transfer_pass": (selected["base_gate_pass"] and
                                        candidate["base_gate_pass"] and
                                        all(abs(value) <= .03 for value in deltas.values())),
        })
    report = {
        "schema": "robot-learning-lab.cove50-hand-mjlab-squat-transfer.v1",
        "plan_sha256": sha(PLAN),
        "selected_pointer_sha256": plan["selected_pointer_sha256"],
        "selected_body_sha256": plan["selected_body_sha256"],
        "candidate_body_sha256": plan["candidate_body_sha256"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "remote_output_hashes_sha256": sha(TRIALS / "remote-sha256.txt"),
        "cases": rows, "comparison": comparison,
        "selected_fresh_baseline_passes": sum(c["selected_gate_pass"] for c in comparison),
        "candidate_transfer_passes": sum(c["candidate_transfer_pass"] for c in comparison),
        "limits": [
            "This zero-shot squat replay does not demonstrate a floor-entry handoff, grasp, lift or possession.",
            "The narrow-bridge CAD/material/bearing assumptions remain unselected and unmeasured.",
            "The two seeds do not cover the full mass, grip, voltage and delay variance envelope.",
        ],
    }
    (OUT / "summary-matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "comparison-matrix.json").write_text(json.dumps({"cases": comparison}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_fresh_baseline_passes": report["selected_fresh_baseline_passes"],
                      "candidate_transfer_passes": report["candidate_transfer_passes"],
                      "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
