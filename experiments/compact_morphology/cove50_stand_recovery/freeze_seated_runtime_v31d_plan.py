"""Freeze the environment-only V31d sentinel plan and fresh-host creation manifest.

This is a local file operation. It never provisions compute. Existing frozen
files are accepted only when their bytes are exactly what this builder emits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from create_seated_runtime_v31d import (
    ATTESTATION_SCHEMA, CAP_S, GPU, IMAGE, LABEL, MANIFEST, MANIFEST_SCHEMA,
    MAX_POD_S, MAX_SPEND, PLAN, PLAN_SCHEMA, RATE, READINESS_S, ROOT, STATE,
    STOP_RESERVE_S, EXPECTED_REFERENCE_SHA, REQUIRED_SOURCE_PATHS,
)


SOURCE_PATHS = {
    "design_pointer": "robot/ape-current-design.json",
    "body": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "checkpoint": "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25/checkpoints/model_25.pt",
    "actuator_catalog": "robot/actuator-catalog.yaml",
    "upright_endpoint_screen": "artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json",
    "upright_left_mirror_endpoint": "artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json",
    "task": "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
    "frame50": "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    "trainer": "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py",
    "preflight": "experiments/compact_morphology/cove50_stand_recovery/preflight_seated_runtime_v31d.py",
    "bootstrap": "experiments/compact_morphology/cove50_stand_recovery/bootstrap_seated_runtime_v31b.sh",
    "runtime_inputs": "experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt",
    "pollen_lock": ".venv-control/src/microduck_rl/uv.lock",
    "upstream_mdp": ".venv-control/src/microduck_rl/src/mjlab_microduck/tasks/mdp.py",
    "bam_mjlab_reference": ".venv-control/lib/python3.12/site-packages/bam/mjlab.py",
    "v30_evaluator": "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_collection_v30.py",
    "watchdog": "experiments/compact_morphology/cove50_stand_recovery/create_seated_runtime_v31d.py",
}


def canonical(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Frozen file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> None:
    if any(SOURCE_PATHS.get(name) != rel
           for name, rel in REQUIRED_SOURCE_PATHS.items()):
        raise RuntimeError("V31d source builder and guard disagree")
    for rel in SOURCE_PATHS.values():
        if not (ROOT / rel).is_file():
            raise RuntimeError(f"Required source is missing: {rel}")
    source_sha = {name: sha(ROOT / rel) for name, rel in SOURCE_PATHS.items()}
    if any(source_sha.get(name) != value
           for name, value in EXPECTED_REFERENCE_SHA.items()):
        raise RuntimeError("V31d pinned historical reference identity changed")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "frozen_before_paid_environment_preflight",
        "purpose": "Verify the exact Cove seated CUDA runtime on one disjoint diagnostic sentinel before a new collection method.",
        "claim_limit": "Environment readiness only. No scored originals, learned policy score, optimizer update, or mechanical qualification.",
        "compiled_control_body": SOURCE_PATHS["body"],
        "mechanical_pointer_status": "rejected_no_selected_mechanical_design; Direct Frame is an experimental control body",
        "source_paths": SOURCE_PATHS,
        "source_sha256": source_sha,
        "device": "cuda:0",
        "sentinel_seed": 34000,
        "sentinel_env_steps_max": 1,
        "scored_original_visits_max": 0,
        "optimizer_updates_max": 0,
        "frozen_score_exclusions": [
            "V30 seed 33001 scripted original consumed by constructor failure; no V30 retries",
            "V25 held seeds 2240-2251",
            "V25 action-free and rare sole-gap tails",
        ],
        "phase_order": [
            "arm local guard and inspect healthy launchd/caffeinate state",
            "create exact uniquely named secure RTX4090 host under request-time cap",
            "attest image, dynamic pod ID, rate, live guard and copied source inputs",
            "run source-bound no-evaluation bootstrap to install reconciled native dependencies",
            "attest full installed dependency closure, source origins and exact package APIs without CUDA task construction",
            "independent live-host review of attestation before CUDA diagnostic",
            "construct exact task on sentinel 34000, reset and record first vectors/action without env.step",
            "at most one explicitly unscored sentinel env.step to exercise CUDA/contact kernels",
            "sync evidence and confirm provider stopped",
        ],
        "runtime_inputs_kind": "pinned install inputs; actual installed closure must be recorded and verified on host",
        "max_pod_wall_s": MAX_POD_S,
        "gpu_listed_rate_usd_per_hour": RATE,
        "gpu_listed_spend_cap_usd": MAX_SPEND,
        "max_start_readiness_s": READINESS_S,
        "stop_reserve_s": STOP_RESERVE_S,
        "watchdog_pod_image": IMAGE,
        "watchdog_gpu_id": GPU,
        "watchdog_service": LABEL,
        "watchdog_state_path": str(STATE.relative_to(ROOT)),
        "watchdog_pod_id": None,
        "cost_limit_note": "Local macOS launchd/caffeinate guard stops within 35 minutes and before $0.50 listed GPU estimate from the pre-create request, with a 60-second stop reserve; no provider-side cap. Storage and posted billing are separate.",
        "stop_conditions": [
            "source or installed dependency closure mismatch",
            "CUDA task constructor, reset, observation, action, or sentinel step failure",
            "local watchdog, provider host, readiness, or listed-rate fault",
            "elapsed time or listed GPU spend cap",
        ],
    }
    plan_bytes = canonical(plan)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "plan_path": str(PLAN.relative_to(ROOT)),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "guard_source_path": SOURCE_PATHS["watchdog"],
        "guard_source_sha256": source_sha["watchdog"],
        "attestation_schema": ATTESTATION_SCHEMA,
        "image": IMAGE,
        "gpu_id": GPU,
        "cloud_type": "SECURE",
        "gpu_count": 1,
        "listed_rate_usd_per_hour": RATE,
        "max_pod_wall_s": MAX_POD_S,
        "gpu_listed_spend_cap_usd": MAX_SPEND,
        "effective_cap_s": CAP_S,
        "stop_reserve_s": STOP_RESERVE_S,
        "max_start_readiness_s": READINESS_S,
        "scored_original_visits_max": 0,
        "sentinel_seed": 34000,
        "container_disk_in_gb": 30,
        "volume_in_gb": 10,
        "ssh": True,
        "ports": ["22/tcp"],
    }
    freeze(PLAN, plan_bytes)
    freeze(MANIFEST, canonical(manifest))
    print(json.dumps({"plan": str(PLAN.relative_to(ROOT)),
                      "plan_sha256": sha(PLAN),
                      "manifest": str(MANIFEST.relative_to(ROOT)),
                      "manifest_sha256": sha(MANIFEST),
                      "guard_source_sha256": source_sha["watchdog"]}, sort_keys=True))


if __name__ == "__main__":
    main()
