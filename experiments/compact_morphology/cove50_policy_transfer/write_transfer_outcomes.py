"""Record a per-checkpoint Cove disposition backed by retained native rollouts.

This is an audit of the selected design, not an update to the historical skill
registry. A simulated motion is not promoted to a physical robot skill here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-transfer-20261008"

# Statuses describe the *Cove* transfer, not the historical checkpoint.
# Every evidence path is checked before writing the audit.
CASES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "forward_walk_stop": ("limited_simulation", "Retrained forward/stop; nominal rollout moves and stops, but path drift, rated-effort saturation, and held-out sensitivity prevent robust qualification.", ("varied150-matrix.json", "robust10v200-fixed-matrix.json")),
    "backward_walk": ("failed_native_replay", "Mapped historical actor fell on Cove; new reverse-command PPO and reserved replays did not meet the reverse gait gate.", ("backward_walk-mapped-seed144.json", "backward-varied300-matrix.json")),
    "backward_stop_candidate": ("failed_native_replay", "Mapped historical stop actor fell on Cove; zero-command behavior was judged with the retrained reverse actor.", ("backward_stop_candidate-mapped-seed144.json", "backward-varied300-matrix.json")),
    "stand_candidate": ("rejected_inferior_baseline", "Retained and retrained learned stand saturate the legs and underperform the passive zero-action Cove stand.", ("stand-loweffort198-seed144.json", "stand-zero-seed144.json")),
    "squat_recover": ("failed_native_replay", "Legal flat-sole and counterbalanced squat PPO did not execute a high-low-high cycle on reserved seeds.", ("squat-counterbalance-physics-probe.json", "squat-counterbalance298-posture-arm-matrix.json")),
    "both_arms_raise_lower": ("motion_only", "Raise and lower commands move both hands while supported, but shoulder rated effort is reached through the hold.", ("arm150-posture-arm-matrix.json", "arm-zero-seed144.json")),
    "independent_arms_v3": ("motion_only", "Independent commands move the intended hand and return; command bleed and shoulder effort remain unqualified.", ("arm150-posture-arm-matrix.json",)),
    "right_foot_hold_from_pose": ("failed_native_replay", "Fresh right stance replays fell within a second.", ("hold-right-varied149-seed144.json", "hold-right-strong99-seed144.json")),
    "left_foot_hold_from_pose": ("failed_native_replay", "Fresh left stance replays fell within a second.", ("hold-left-varied149-seed144.json", "hold-left-varied149-seed905.json")),
    "weight_shift_both_sides": ("failed_native_replay", "Mapped two-sided shift did not produce sustained single support.", ("weight-shift-mapped-left-seed144.json", "weight-shift-mapped-right-seed144.json")),
    "turn_curve_500_candidate": ("failed_native_replay", "Planar continuation survives but curves drift and rate-limited legs dominate.", ("planar200v3-planar-matrix.json",)),
    "turn_700_candidate": ("failed_native_replay", "Direct mapped turn misses requested yaw and saturates legs; planar continuation also misses turning commands.", ("turn_700_candidate-mapped-seed144.json", "planar200v3-planar-matrix.json")),
    "right_hand_reach_candidate": ("failed_native_replay", "Mapped and retrained reach did not contact the floor toy on fresh seeds; legacy target has a measured upright-pose geometry gap.", ("reach-mapped-seed144.json", "skill-geometry-audit.json", "reach-counterbalance-final-seed144.json", "reach-counterbalance-final-seed905.json")),
    "jaw_contact_lift_candidate": ("blocked_invalid_handoff", "Floor-toy handoff assumed a reachable pregrasp; Cove reach gate failed.", ("skill-geometry-audit.json", "reach-varied99-seed144.json")),
    "held_lift_pregrasp_v5": ("blocked_invalid_handoff", "Historical held-cube reset and jaw placement are not a demonstrated Cove pregrasp.", ("skill-geometry-audit.json", "reach-varied99-seed144.json")),
    "open_jaw_acquire_v5": ("blocked_invalid_handoff", "Historical acquire scene requires a Cove hand-to-cube handoff that has not passed.", ("skill-geometry-audit.json", "reach-varied99-seed144.json")),
    "place_release_v5": ("blocked_invalid_handoff", "Place/release depends on a reachable held-cube reset and qualified acquire stage.", ("skill-geometry-audit.json", "reach-varied99-seed144.json")),
    "place_after_acquire_v5": ("blocked_invalid_handoff", "Composed acquire/place cannot be claimed while Cove acquisition and held-cube reset are invalid.", ("skill-geometry-audit.json", "reach-varied99-seed144.json")),
    "goal_conditioned_place_candidate_v5": ("blocked_invalid_handoff", "Goal-conditioned place has no qualified Cove acquire/held-cube handoff.", ("skill-geometry-audit.json", "reach-varied99-seed144.json")),
    "one_foot_hold_right_full_collision": ("failed_native_replay", "Stronger historical full-collision source and its Cove continuation fell on fresh replays.", ("hold-right-fullcollision-mapped-seed144.json", "hold-right-strong99-seed144.json")),
    "one_foot_hold_left_full_collision": ("failed_native_replay", "Mapped upright-clear left source and Cove continuation fell on fresh replays.", ("hold-left-upright-mapped-seed144.json", "hold-left-varied149-seed144.json")),
    "foot_lift_left_reference_tracking": ("blocked_invalid_reference", "Authored reference exceeds Cove joint limits; native tracking requires a newly retargeted feasible motion.", ("skill-geometry-audit.json",)),
    "foot_lift_right_reference_tracking": ("blocked_invalid_reference", "Authored reference exceeds Cove joint limits; native tracking requires a newly retargeted feasible motion.", ("skill-geometry-audit.json",)),
    "step_left_reference_tracking": ("blocked_invalid_reference", "Step depends on the invalid source foot-transition reference and a single-support gate that failed.", ("skill-geometry-audit.json", "hold-left-varied149-seed144.json")),
    "step_right_reference_tracking": ("blocked_invalid_reference", "Step depends on the invalid source foot-transition reference and a single-support gate that failed.", ("skill-geometry-audit.json", "hold-right-strong99-seed144.json")),
    "step_left_commanded_from_lift": ("blocked_invalid_reference", "Commanded touchdown starts from a raised-foot handoff not qualified on Cove.", ("skill-geometry-audit.json", "hold-left-varied149-seed144.json")),
    "step_left_distilled": ("blocked_invalid_reference", "Distilled step depends on a feasible Cove lift/touchdown teacher, absent after reference and support failures.", ("skill-geometry-audit.json", "hold-left-varied149-seed144.json")),
    "step_right_distilled": ("blocked_invalid_reference", "Distilled step depends on a feasible Cove lift/touchdown teacher, absent after reference and support failures.", ("skill-geometry-audit.json", "hold-right-strong99-seed144.json")),
    "cad_v8_forward_walk": ("failed_native_replay", "Mapped 22-to-21-axis CAD gait fell in all four native Cove worlds.", ("cad-v8-forward-mapped-seed144.json",)),
}

TRAINED_CANDIDATES = {
    "forward_walk_stop": ("cove50-forward-varied150.pt", "checkpoints/robust-forward/model_845.pt"),
    "backward_walk": ("checkpoints/backward/model_299.pt",),
    "backward_stop_candidate": ("checkpoints/backward/model_299.pt",),
    "stand_candidate": ("cove50-stand-loweffort198.pt",),
    "squat_recover": ("checkpoints/squat/model_298.pt",),
    "both_arms_raise_lower": ("cove50-arms-varied149.pt",),
    "independent_arms_v3": ("cove50-independent-arms-varied149.pt",),
    "right_foot_hold_from_pose": ("cove50-hold-right-varied149.pt",),
    "left_foot_hold_from_pose": ("cove50-hold-left-varied149.pt",),
    "turn_curve_500_candidate": ("cove50-planar-varied200.pt",),
    "turn_700_candidate": ("cove50-planar-varied200.pt",),
    "right_hand_reach_candidate": ("cove50-reach-varied99.pt", "checkpoints/reach/model_198.pt"),
    "one_foot_hold_right_full_collision": ("checkpoints/right-hold/model_99.pt",),
    "one_foot_hold_left_full_collision": ("cove50-hold-left-varied149.pt",),
}


def main() -> None:
    inventory = json.loads((OUT / "policy-inventory.json").read_text())
    ledger = json.loads((OUT / "body-ledger.json").read_text())
    if inventory["current_design"] != ledger["selected_design_id"]:
        raise ValueError("Inventory and selected body point to different designs")
    ids = {item["id"] for item in inventory["policies"]}
    if ids != set(CASES):
        raise ValueError(f"Inventory mismatch: missing={ids - set(CASES)}, extra={set(CASES) - ids}")
    rows = []
    for item in inventory["policies"]:
        status, finding, evidence = CASES[item["id"]]
        for name in evidence:
            if not (OUT / name).is_file():
                raise FileNotFoundError(OUT / name)
        rows.append({
            "id": item["id"],
            "source_checkpoint": item["checkpoint"],
            "source_checkpoint_sha256": item["checkpoint_sha256"],
            "cove_status": status,
            "finding": finding,
            "evidence": [str((OUT / name).relative_to(ROOT)) for name in evidence],
            "trained_candidates": [
                {"checkpoint": str((OUT / name).relative_to(ROOT)),
                 "sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest()}
                for name in TRAINED_CANDIDATES.get(item["id"], ())
            ],
        })
    output = {
        "schema": "robot-learning-lab.cove50-policy-outcomes.v1",
        "selected_design_id": ledger["selected_design_id"],
        "selected_design_pointer_sha256": ledger["selected_design_sha256"],
        "training_body_sha256": hashlib.sha256((OUT / "cove50-direct-training.xml").read_bytes()).hexdigest(),
        "policy_count": len(rows),
        "qualified_count": sum(row["cove_status"] == "qualified" for row in rows),
        "qualification": "No historical skill is qualified for physical use by these provisional simulation results.",
        "policies": rows,
    }
    (OUT / "transfer-outcomes.json").write_text(json.dumps(output, indent=2) + "\n")
    print(OUT / "transfer-outcomes.json")


if __name__ == "__main__":
    main()
