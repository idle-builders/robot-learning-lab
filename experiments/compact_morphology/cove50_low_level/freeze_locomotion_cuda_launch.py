"""Freeze the source-only locomotion CUDA launch plan and guarded pod manifest.

This is a local file operation. It never arms a guard or provisions compute.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from create_locomotion_probe_guard import (
    ART, ATTESTATION_SCHEMA, BUNDLE, CAP_S, EXPECTED_REFERENCE_SHA,
    GPU, IMAGE, LABEL, MANIFEST, MANIFEST_SCHEMA, MAX_POD_S, MAX_SPEND,
    PLAN, PLAN_SCHEMA, RATE, READINESS_S, REQUIRED_SOURCE_PATHS, ROOT,
    STATE, STOP_RESERVE_S,
)


EXPECTED_CLOSURE_SHA = "7c2f52255fd27e42d2d6153b61d05c9eb7b05116a8134b65541f53451cefdfef"
POLLEN = ROOT / ".venv-control/src/microduck_rl"
POLLEN_COMMIT = "0bf989701f14d4008c866ffd6f09067870f99379"
BUNDLE_ENTRIES = (
    "robot/ape-current-design.json",
    "robot/actuator-catalog.yaml",
    "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt",
    "artifacts/ape-cove50-turn-recovery-20261008/turneffort25-model950.pt",
    "experiments/compact_morphology/mjlab_ape",
    "experiments/compact_morphology/cove50_low_level/run_signed_stop_probe.py",
    "experiments/compact_morphology/cove50_low_level/preflight_locomotion_cuda.py",
    "experiments/compact_morphology/cove50_low_level/create_locomotion_probe_guard.py",
    "experiments/compact_morphology/cove50_stand_recovery/bootstrap_seated_runtime_v31b.sh",
    "experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt",
    "experiments/compact_morphology/cove50_stand_recovery/preflight_seated_runtime_v31d.py",
    ".venv-control/src/microduck_rl",
    ".venv-control/lib/python3.12/site-packages/bam/mjlab.py",
    "artifacts/ape-cove50-locomotion-audit-20261014/cuda-launch/plan.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def freeze(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"Frozen file differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)


def clean_pollen() -> None:
    head = subprocess.check_output(["git", "-C", str(POLLEN), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(POLLEN), "status", "--porcelain"], text=True).strip()
    if head != POLLEN_COMMIT or dirty:
        raise RuntimeError("Pinned Pollen checkout changed or is dirty")


def tar_filter(item: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(item.name).parts
    if any(part.startswith("._") or part in (".DS_Store", "__pycache__", ".pytest_cache")
           for part in parts):
        return None
    if item.issym() and (item.linkname.startswith("/") or ".." in Path(item.linkname).parts):
        raise RuntimeError(f"Unsafe source symlink: {item.name}")
    item.mtime = 0
    item.uid = item.gid = 0
    item.uname = item.gname = ""
    return item


def make_bundle() -> bytes:
    memory = io.BytesIO()
    with gzip.GzipFile(fileobj=memory, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for relative in BUNDLE_ENTRIES:
                path = ROOT / relative
                if not path.exists():
                    raise RuntimeError(f"Bundle source missing: {relative}")
                archive.add(path, arcname=relative, recursive=True, filter=tar_filter)
    return memory.getvalue()


def main() -> None:
    clean_pollen()
    source_paths = dict(REQUIRED_SOURCE_PATHS)
    source_sha = {name: sha(ROOT / relative) for name, relative in source_paths.items()}
    if any(source_sha[name] != expected for name, expected in EXPECTED_REFERENCE_SHA.items()):
        raise RuntimeError("Historical source identity changed")
    if os.environ.get("COPYFILE_DISABLE") != "1":
        raise RuntimeError("Set COPYFILE_DISABLE=1 before source packaging")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "frozen_before_paid_opened_diagnostic",
        "purpose": "One-world CUDA entry and 18 opened native signed-motion/stop replay starts; zero optimizer updates.",
        "claim_limit": "Runtime and opened diagnosis only, not motor-feasible qualification or physical transfer.",
        "compiled_control_body": source_paths["body"],
        "mechanical_pointer_status": "rejected_no_selected_mechanical_design; Direct Frame experimental control body",
        "source_paths": source_paths,
        "source_sha256": source_sha,
        "pollen_commit": POLLEN_COMMIT,
        "required_installed_closure_content_sha256": EXPECTED_CLOSURE_SHA,
        "device": "cuda:0",
        "sentinel_seeds": [46998, 46999],
        "sentinel_env_steps_max": 2,
        "opened_probe_seeds": list(range(47001, 47019)),
        "opened_native_visits_max": 18,
        "optimizer_updates_max": 0,
        "worlds": 1,
        "frozen_score_exclusions": [
            "V30 failed original seed 33001",
            "V25 held seeds 2240-2251",
            "V25 action-free and rare sole-gap tails",
            "All other sealed skill score reserves",
        ],
        "phase_order": [
            "verify frozen source bundle, local contract, provider rate and healthy launchd/caffeinate guard",
            "create one uniquely named secure RTX4090 pod under request-time cost cap",
            "attest exact provider host and copy bundle with AppleDouble-free tar verification",
            "bootstrap reconciled V31d packages, pip check and 227-distribution closure",
            "independent live-host review of source and runtime attestation before CUDA",
            "run two disjoint exact evaluator one-step CUDA sentinels",
            "refresh live guard attestation and run at most 18 opened one-world diagnostics",
            "sync reports/logs/traces, stop, confirm provider state and late-create discovery",
        ],
        "stop_conditions": [
            "source, bundle, package closure or imported module origin mismatch",
            "guard, provider GPU/image/rate, readiness or deadline fault",
            "CUDA constructor/reset/action/one-step sentinel fault",
            "native case timeout, missing report, nonfinite action or output mismatch",
            "18 opened visits, zero optimizer updates, or elapsed cost cap",
        ],
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
        "cost_limit_note": "Local launchd/caffeinate guard: max 40 minutes, listed $0.60 RTX4090 estimate with 60 s reserve; no provider-side cap or verified posted bill.",
    }
    freeze(PLAN, canonical(plan))
    bundle = make_bundle()
    freeze(BUNDLE, bundle)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "plan_path": str(PLAN.relative_to(ROOT)),
        "plan_sha256": sha(PLAN),
        "guard_source_path": source_paths["watchdog"],
        "guard_source_sha256": source_sha["watchdog"],
        "bundle_path": str(BUNDLE.relative_to(ROOT)),
        "bundle_sha256": sha(BUNDLE),
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
        "opened_native_visits_max": 18,
        "worlds": 1,
        "container_disk_in_gb": 30,
        "volume_in_gb": 10,
        "ssh": True,
        "ports": ["22/tcp"],
    }
    freeze(MANIFEST, canonical(manifest))
    print(json.dumps({"plan_sha256": sha(PLAN), "bundle_sha256": sha(BUNDLE),
                      "bundle_bytes": BUNDLE.stat().st_size,
                      "manifest_sha256": sha(MANIFEST)}, sort_keys=True))


if __name__ == "__main__":
    main()
