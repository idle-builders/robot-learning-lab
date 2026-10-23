"""Freeze a source-checked, unscored CUDA preflight packet. Never provisions GPU."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3d"
METHOD = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/method.json"
AMENDMENT = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3c/plan.json"
POLLEN = ROOT / ".venv-control/src/microduck_rl"
POLLEN_COMMIT = "0bf989701f14d4008c866ffd6f09067870f99379"
SOURCE_ENTRIES = (
    "robot/ape-current-design.json",
    "robot/actuator-catalog.yaml",
    "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt",
    "artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json",
    "artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json",
    "artifacts/ape-cove50-locomotion-learning-v3-20261014/method.json",
    "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3c/plan.json",
    "artifacts/ape-cove50-locomotion-audit-20261014/cuda-launch-v2/plan.json",
    "experiments/compact_morphology/mjlab_ape",
    "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py",
    "experiments/compact_morphology/cove50_low_level/preflight_locomotion_full_state_v3b.py",
    "experiments/compact_morphology/cove50_low_level/preflight_locomotion_entry_v3c.py",
    "experiments/compact_morphology/cove50_low_level/checkout_import_gate_v2.py",
    "experiments/compact_morphology/cove50_low_level/run_locomotion_preflight_v3d.py",
    "experiments/compact_morphology/cove50_low_level/freeze_locomotion_preflight_v3d.py",
    "experiments/compact_morphology/cove50_stand_recovery/bootstrap_seated_runtime_v31b.sh",
    "experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt",
    "experiments/compact_morphology/cove50_stand_recovery/preflight_seated_runtime_v31d.py",
    ".venv-control/src/microduck_rl",
    ".venv-control/lib/python3.12/site-packages/bam/mjlab.py",
)
EXPECTED = {
    "robot/ape-current-design.json": "363ecd1effce0795f64b03066d8a17b90ba81d1036496d78eee9026762466b41",
    "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml": "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c",
    "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt": "1d01d490e3754bf9cb14e19588d784cb7bd1c9d6e82ae0b216c3ce06590e27a2",
    "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py": "3b518e3dae8b53df7383252299d8140b3b766be2eb329dde95991bd595116c4d",
    "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py": "7f86d8ec4c700fd1db38ad7989b7c37fe30a153946975c6f9440bcc6e7eff3e2",
    "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py": "e6e0bd930001f01c6840fb6700ed7dae7dc98377425e64e0687a37eb2faed203",
    "experiments/compact_morphology/mjlab_ape/evaluate.py": "1573c96e54ab4e5be9abd9851f11dda130a6d77eec1cfa8cf886b9cd76fc06fc",
    "artifacts/ape-cove50-locomotion-learning-v3-20261014/method.json": "f33e9732560d9533ddab6346138d1d82be337019123842248730216031eb0af7",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def files() -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for entry in SOURCE_ENTRIES:
        path = ROOT / entry
        if not path.exists():
            raise RuntimeError(f"Missing packet input: {entry}")
        members = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for member in members:
            rel = member.relative_to(ROOT).as_posix()
            if any(part in ("__pycache__", ".pytest_cache", ".git") for part in member.parts):
                continue
            if member.name.startswith("._") or member.name == ".DS_Store":
                continue
            if member.is_symlink() and not member.resolve().is_relative_to(ROOT):
                # The required left endpoint is an artifact-store symlink: the
                # dereferenced content is verified below and packed as bytes.
                if rel != "artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json":
                    raise RuntimeError(f"Unexpected external symlink: {rel}")
            result[rel] = member.read_bytes()
    for rel, expected in EXPECTED.items():
        if sha(result[rel]) != expected:
            raise RuntimeError(f"Frozen input changed: {rel}")
    return result


def main() -> None:
    if os.environ.get("COPYFILE_DISABLE") != "1":
        raise RuntimeError("Set COPYFILE_DISABLE=1 to avoid AppleDouble bundle files")
    head = subprocess.check_output(["git", "-C", str(POLLEN), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(POLLEN), "status", "--porcelain"], text=True).strip()
    if head != POLLEN_COMMIT or dirty:
        raise RuntimeError("Pinned upstream Pollen checkout changed or is dirty")
    content = files()
    amendment = json.loads(AMENDMENT.read_text())
    if (amendment["amends_without_rewriting"]["v3_method_sha256"] != sha(METHOD.read_bytes())
            or amendment["height_and_case_protocol"]["primary_evaluator_cli"] != "--initial-root-z 0.231"):
        raise RuntimeError("V3c amendment contract changed")
    packet = {
        "schema": "robot-learning-lab.locomotion-v3d-cuda-preflight-packet.v1",
        "status": "frozen_local_packet_no_paid_launch",
        "claim_limit": "One unscored CUDA full-state and actual-evaluator command-switch sentinel; zero V3 originals, zero optimizer updates, no score or skill qualification.",
        "body": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
        "checkpoint": "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt",
        "source_sha256": {rel: sha(data) for rel, data in sorted(content.items())},
        "pollen_commit": POLLEN_COMMIT,
        "runtime_closure_sha256": "7c2f52255fd27e42d2d6153b61d05c9eb7b05116a8134b65541f53451cefdfef",
        "worker_root": "/root/robot-learning-lab",
        "worker_pythonpath": "/root/robot-learning-lab/experiments/compact_morphology/mjlab_ape/src:/root/robot-learning-lab",
        "cuda_device": "cuda:0",
        "sentinel_seed": 47997,
        "development_seeds_excluded": list(range(48001, 48019)),
        "sealed_score_seeds_excluded": list(range(49001, 49025)),
        "max_worlds": 1,
        "max_native_steps": 202,
        "max_optimizer_updates": 0,
        "evaluator_args": ["--task", "Mjlab-Velocity-Flat-Ape-Elbows", "--physics-model", "frame50", "--initial-root-z", "0.231", "--yaw-only-head", "--arms-down", "--worlds", "1", "--seed", "47997", "--velocity-x", "0.35", "--velocity-y", "0", "--angular-z", "0", "--motor-rail-v", "12", "--command-delay-ms", "0", "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002", "--stop-at-s", "4.0", "--duration-s", "4.04"],
        "provider_guard_required": {
            "create_permitted_by_packet": False,
            "requires_reviewed_local_launchd_provider_stop_guard": True,
            "max_paid_wall_s": 900,
            "max_listed_gpu_spend_usd_at_0p89_per_hour": 0.25,
            "stop_reserve_s": 90,
            "latest_guard_attestation_age_s": 30,
            "note": "An attestation file alone cannot stop a pod. Do not create one until a reviewed provider-state-checking stop guard is armed and independently checked."
        },
        "abort_on": ["missing or changed source/asset/package closure", "wrong checkout import origin", "wrong GPU or missing fresh provider stop guard", "wrong checkpoint resolver/full actor/critic/optimizer/normalizer/iteration", "mismatched paired first physical state/observation/action/value", "nonfinite state/action/contact", "actual evaluate.py CLI lacks move-to-zero step at 4.00s", "missing or invalid native report/trace", "time/cost reserve insufficient"],
    }
    ART.mkdir(parents=True, exist_ok=True)
    packet_data = (json.dumps(packet, indent=2, sort_keys=True) + "\n").encode()
    packet_path = ART / "packet.json"
    if packet_path.exists() and packet_path.read_bytes() != packet_data:
        raise RuntimeError("Frozen packet differs; create a new version")
    packet_path.write_bytes(packet_data)
    content[packet_path.relative_to(ROOT).as_posix()] = packet_data
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for rel, data in sorted(content.items()):
                item = tarfile.TarInfo(rel)
                item.size, item.mode, item.mtime = len(data), 0o644, 0
                item.uid = item.gid = 0
                tar.addfile(item, io.BytesIO(data))
    bundle = ART / "worker-source.tar.gz"
    if bundle.exists() and bundle.read_bytes() != buffer.getvalue():
        raise RuntimeError("Frozen worker bundle differs; create a new version")
    bundle.write_bytes(buffer.getvalue())
    manifest = {"schema": "robot-learning-lab.locomotion-v3d-local-bundle.v1",
                "packet_sha256": sha(packet_data), "bundle_sha256": sha(bundle.read_bytes()),
                "bundle_bytes": bundle.stat().st_size,
                "source_files": len(content) - 1,
                "status": "locally_bundled_not_run_on_cuda"}
    manifest_path = ART / "bundle-manifest.json"
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_data:
        raise RuntimeError("Frozen bundle manifest differs")
    manifest_path.write_bytes(manifest_data)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
