"""Check exact frozen cardinal source and optionally its CUDA runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v2/method.json"
PACKAGES = {"torch": "2.9.1+cu128", "triton": "3.5.1", "warp-lang": "1.12.0",
            "mjlab": "1.3.0", "rsl-rl-lib": "5.0.1", "mujoco": "3.13.0",
            "mujoco-warp": "3.8.1", "numpy": "2.4.1"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    if (method["schema"] != "robot-learning-lab.cove50-cardinal-v2.method.v2"
            or method["learner"]["worlds"] != 1024
            or method["learner"]["additional_updates"] != 40
            or len(method["development_cases"]) != 8):
        raise RuntimeError("Frozen cardinal method differs")
    for item in method["source"].values():
        if not isinstance(item, dict) or "path" not in item:
            continue
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen cardinal source differs: {path}")
    report = {"method_sha256": sha(METHOD), "source_files": len(method["source"]),
              "python": sys.version.split()[0]}
    if args.runtime:
        actual = {name: importlib.metadata.version(name) for name in PACKAGES}
        if actual != PACKAGES:
            raise RuntimeError(f"Installed runtime differs: {actual}")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one CUDA GPU")
        report.update(packages=actual, gpu=torch.cuda.get_device_name(0))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
