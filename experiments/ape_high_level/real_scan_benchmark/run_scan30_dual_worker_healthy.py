"""Frozen healthy Scan30 policy on the dual-depth experiment worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_scan30_unified_policy as unified


ROOT, ART = unified.ROOT, unified.ART
OUT = ROOT / "assets/real_indoor_scenes/scan30_dual_worker_healthy_20261015"
PRECOMMIT = ART / "scan30_dual_worker_healthy_precommit.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    baseline = ART / "scan30_unified_policy_precommit.json"
    paired = ART / "scan30_dual_rgbd_route_precommit.json"
    frozen = {"schema": "ape.scan30-dual-worker-healthy-precommit.v1",
        "policy_precommit": str(baseline.relative_to(ROOT)),
        "policy_precommit_sha256": sha(baseline),
        "paired_dual_precommit": str(paired.relative_to(ROOT)),
        "paired_dual_precommit_sha256": sha(paired),
        "wrapper_source": str(Path(__file__).relative_to(ROOT)),
        "wrapper_sha256": sha(Path(__file__)),
        "cases": [{"scene_key": c["scene_key"],
                   "capture_group": c["capture_group"]}
                  for c in unified.cohort()],
        "role": "Unmodified from-start Scan30 reference on the same A40 Python/GPU worker as the prospective dual RGB-D route arm. No room-specific changes."}
    if PRECOMMIT.is_file():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen worker baseline changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    unified.OUT = OUT
    unified.BASE_OUT = OUT / "base"
    unified.SHOW_OUT = OUT / "show"
    unified.REPORT = ART / "scan30_dual_worker_healthy_report.json"
    unified.main()


if __name__ == "__main__":
    main()
