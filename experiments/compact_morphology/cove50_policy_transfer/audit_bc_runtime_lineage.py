"""Record the saved BC actors' native control lineage without replaying them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INTERFACE = ROOT / "artifacts/ape-policy-interface-audit-20261012/report.json"
POINTER = ROOT / "robot/ape-current-design.json"
REGISTRY = ROOT / "skills/ape/registry.json"
OUT = ROOT / "artifacts/ape-policy-bc-lineage-20261013"
BC_IDS = (
    "held_lift_pregrasp_v5",
    "open_jaw_acquire_v5",
    "place_release_v5",
    "place_after_acquire_v5",
    "goal_conditioned_place_candidate_v5",
)
SOURCE_FILES = (
    "experiments/compact_morphology/mjlab_ape/src/ape_walk/reach.py",
    "experiments/compact_morphology/mjlab_ape/src/ape_walk/grasp.py",
    "experiments/compact_morphology/mjlab_ape/src/ape_walk/place_tracking.py",
    "experiments/compact_morphology/mjlab_ape/collect_jaw_demos.py",
    "experiments/compact_morphology/mjlab_ape/collect_place_demos.py",
    "experiments/compact_morphology/mjlab_ape/collect_place_from_acquire.py",
    "experiments/compact_morphology/mjlab_ape/train_jaw_bc.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    interface = json.loads(INTERFACE.read_text())
    registry = json.loads(REGISTRY.read_text())
    pointer = json.loads(POINTER.read_text())
    if (interface["pointer_sha256"] != sha(POINTER) or
            interface["registry_sha256"] != sha(REGISTRY) or
            pointer["selected_design_id"] is not None or
            pointer["status"] != "rejected_no_selected_mechanical_design"):
        raise ValueError("Policy shelf or selected-body pointer changed")
    expected = interface["bc_observation_source_sha256"]
    if any(expected[path] != sha(ROOT / path) for path in SOURCE_FILES):
        raise ValueError("BC collector or native task lineage changed")
    policies = {row["id"]: row for row in registry["policies"]}
    if any(policies[identifier]["training_config"]["config.json"]["sha256"] !=
           sha(ROOT / policies[identifier]["training_config"]["config.json"]["local_path"])
           for identifier in BC_IDS):
        raise ValueError("Saved BC config changed")

    # reach_env_cfg sets 0.002 s and decimation 10; grasp and place_tracking
    # inherit that task and modify right-arm action scales. These values are
    # source lineage only: no checkpoint-bound resolved env config was saved.
    rows = []
    for identifier in BC_IDS:
        policy = policies[identifier]
        config = json.loads((ROOT / policy["training_config"]["config.json"]["local_path"]).read_text())
        if (config["train"]["hdf5_normalize_obs"] is not False or
                config["train"]["action_config"]["actions"]["normalization"] is not None or
                config["experiment"]["rollout"]["enabled"] is not False):
            raise ValueError(f"BC normalization or rollout setting changed: {identifier}")
        held = identifier == "held_lift_pregrasp_v5"
        composition = identifier in (
            "place_after_acquire_v5", "goal_conditioned_place_candidate_v5")
        rows.append({
            "id": identifier,
            "saved_task": policy["task"],
            "saved_config_sha256": policy["training_config"]["config.json"]["sha256"],
            "source_implied_physics_dt_s": 0.002,
            "source_implied_decimation": 10,
            "source_implied_control_period_s": 0.02,
            "source_implied_control_rate_hz": 50,
            "right_shoulder_pitch_scale": 0.35,
            "right_elbow_pitch_scale": 0.9,
            "right_gripper_scale": 0.0 if held else 0.5,
            "saved_dataset_obs_normalization": False,
            "saved_dataset_action_normalization": None,
            "native_stage": ("held_jaw" if held else "acquire" if identifier ==
                             "open_jaw_acquire_v5" else "place_after_acquire" if composition
                             else "standalone_place"),
            "stage_offset_handoff_required": composition,
            "qualification": "task_source_implied_not_checkpoint_config_or_measured_runtime",
        })
    report = {
        "schema": "robot-learning-lab.ape-bc-runtime-lineage.v1",
        "role": "rejected_body_source_lineage_no_policy_rollout",
        "source_sha256": {
            "script": sha(Path(__file__)),
            "interface_report": sha(INTERFACE),
            "pointer": sha(POINTER),
            "registry": sha(REGISTRY),
            **{path: sha(ROOT / path) for path in SOURCE_FILES},
        },
        "policy_count": len(rows),
        "cases": rows,
        "interpretation": (
            "The saved BC configs contain no MJLab control period or joint-position "
            "scale. The pinned collector/task source lineage implies 50 Hz and "
            "the listed shoulder, elbow and jaw scales. Place-after-acquire "
            "changes action offsets and observation history at stage handoff. "
            "These are source-inferred conventions, not resolved checkpoint-bound "
            "runtime values, first actions, contact results or selected-body transfer."),
    }
    matrix = {
        "schema": "robot-learning-lab.ape-bc-runtime-lineage-matrix.v1",
        "role": report["role"],
        "source_report": str((OUT / "report.json").relative_to(ROOT)),
        "policy_count": len(rows),
        "cases": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    matrix["source_report_sha256"] = sha(OUT / "report.json")
    (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"Audited BC native source lineage for {len(rows)} saved policies")


if __name__ == "__main__":
    main()
