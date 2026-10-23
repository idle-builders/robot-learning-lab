"""Deterministic, source-checked cardinal CUDA worker bundle."""

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
OUT = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v2"
ENTRIES = (
    "robot/ape-current-design.json", "robot/actuator-catalog.yaml",
    "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt",
    "artifacts/ape-low-level-training-20261014/cardinal-v1/candidate/candidate-final.pt",
    "artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json",
    "artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json",
    "artifacts/ape-low-level-training-20261014/cardinal-v2/method.json",
    "experiments/compact_morphology/mjlab_ape",
    "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py",
    "experiments/compact_morphology/cove50_low_level/preflight_cardinal_resume_v1.py",
    "experiments/compact_morphology/cove50_directional_motion/check_cardinal_v2.py",
    "experiments/compact_morphology/cove50_directional_motion/run_cardinal_v2_sentinel.py",
    "experiments/compact_morphology/cove50_directional_motion/run_cardinal_v2_candidate.py",
    "experiments/compact_morphology/cove50_directional_motion/summarize_cardinal_v2.py",
    "experiments/compact_morphology/cove50_directional_motion/bootstrap_cardinal_v2.sh",
    "experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt",
    ".venv-control/src/microduck_rl",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if os.environ.get("COPYFILE_DISABLE") != "1":
        raise ValueError("Set COPYFILE_DISABLE=1 for AppleDouble-free bundle")
    upstream = ROOT / ".venv-control/src/microduck_rl"
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != "0bf989701f14d4008c866ffd6f09067870f99379":
        raise ValueError("Pinned Pollen source commit changed")
    content = {}
    for entry in ENTRIES:
        path = ROOT / entry
        if not path.exists():
            raise FileNotFoundError(path)
        members = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for member in members:
            if any(part in ("__pycache__", ".pytest_cache", ".git") for part in member.parts):
                continue
            if member.name.startswith("._") or member.name == ".DS_Store":
                continue
            rel = member.relative_to(ROOT).as_posix()
            if member.is_symlink() and not member.resolve().is_relative_to(ROOT.resolve()) and rel != (
                    "artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json"):
                raise ValueError(f"Unreviewed external symlink: {rel}")
            content[rel] = member.read_bytes()
    method = json.loads(content["artifacts/ape-low-level-training-20261014/cardinal-v2/method.json"])
    for source in method["source"].values():
        if isinstance(source, dict) and "path" in source:
            if sha(content[source["path"]]) != source["sha256"]:
                raise ValueError(f"Frozen cardinal source hash mismatch: {source['path']}")
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for rel, data in sorted(content.items()):
                item = tarfile.TarInfo(rel)
                item.size, item.mode, item.mtime = len(data), 0o644, 0
                item.uid = item.gid = 0
                tar.addfile(item, io.BytesIO(data))
    bundle = OUT / "worker-source.tar.gz"
    if bundle.exists() and bundle.read_bytes() != out.getvalue():
        raise ValueError("Cardinal bundle differs; version change")
    bundle.write_bytes(out.getvalue())
    manifest = {"schema": "robot-learning-lab.cardinal-v2.bundle.v1",
                "method_sha256": sha(content["artifacts/ape-low-level-training-20261014/cardinal-v2/method.json"]),
                "bundle_sha256": sha(bundle.read_bytes()), "bundle_bytes": bundle.stat().st_size,
                "source_files": len(content), "pollen_commit": head,
                "status": "source_frozen_before_paid_cuda"}
    path = OUT / "bundle-manifest.json"
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("Cardinal bundle manifest differs")
    path.write_bytes(encoded)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
