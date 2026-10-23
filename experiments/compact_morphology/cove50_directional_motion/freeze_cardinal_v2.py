"""Freeze the independently reviewed equal-step movement pilot before CUDA work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    method = json.loads((OUT / "method-draft.json").read_text())
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["id"] != "cove-50-direct-frame":
        raise RuntimeError("Exact Direct Frame source pointer changed")
    if (method["learner"]["additional_updates"] != 40
            or method["learner"]["expected_environment_steps"] != 983040
            or method["comparison"]["native_visits"] != 24
            or len(method["development_cases"]) != 8):
        raise RuntimeError("Reviewed equal-step pilot differs")
    if method["local_entry"]["full_state_check_sha256"] != sha(OUT / "local-resume-check.json"):
        raise RuntimeError("Corrected 40-update local preflight differs")
    if json.loads((OUT / "local-resume-check.json").read_text())["launch_updates"] != 40:
        raise RuntimeError("Local full-state entry still targets other update count")
    for source in method["source"].values():
        if isinstance(source, dict) and "path" in source:
            if sha(ROOT / source["path"]) != source["sha256"]:
                raise RuntimeError(f"Source hash differs: {source['path']}")
    method["schema"] = "robot-learning-lab.cove50-cardinal-v2.method.v2"
    method["status"] = "method_frozen_before_any_v2_cuda_or_optimizer_visit"
    method["independent_review"] = "artifacts/ape-low-level-training-20261014/cardinal-v2/INDEPENDENT-METHOD-REVIEW.md"
    path = OUT / "method.json"
    payload = (json.dumps(method, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError("Frozen method differs; use another version")
    path.write_bytes(payload)
    print(json.dumps({"method": str(path.relative_to(ROOT)), "sha256": sha(path),
                      "worlds": 1024, "updates": 40, "native_visits": 24}, indent=2))


if __name__ == "__main__":
    main()
