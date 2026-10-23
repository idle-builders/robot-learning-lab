"""Check the frozen V4 source and its installed training runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4/method.json"
PACKAGES = {"torch": "2.9.1+cu128", "triton": "3.5.1", "warp-lang": "1.12.0",
            "mjlab": "1.3.0", "rsl-rl-lib": "5.0.1", "mujoco": "3.13.0",
            "mujoco-warp": "3.8.1", "numpy": "2.4.1"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    if (method.get("schema") != "robot-learning-lab.cove50-locomotion-fast-v4.method.v1"
            or method.get("status") != "method_frozen_before_any_v4_cuda_visit_or_optimizer_update"
            or method["learner"]["worlds"] != 512
            or method["learner"]["additional_updates"] != 24):
        raise RuntimeError("V4 method contract changed")
    for item in method["source"].values():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"V4 source differs: {path}")
    result = {"method_sha256": sha(METHOD), "source_files": len(method["source"]),
              "python": sys.version.split()[0]}
    if args.runtime:
        actual = {name: importlib.metadata.version(name) for name in PACKAGES}
        if actual != PACKAGES:
            raise RuntimeError(f"Installed V4 runtime differs: {actual}")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("V4 requires one CUDA GPU")
        result["packages"] = actual
        result["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
