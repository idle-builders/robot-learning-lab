"""Local prebundle audit of eager ape task imports and their endpoint bytes."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
APE = ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk"
ART = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014/cuda-launch-v2"
OUTPUT = ART / "import-closure-audit.json"
EAGER_MODULES = ("reach", "grasp", "grasp_tracking", "place_tracking",
                 "place_stages", "foot_upright")
ENDPOINTS = {
    "upright_endpoint_screen": (
        "artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json",
        "e954dd92a0b16e8299b684d0bc3e6891f7743b576b661e71bb72d9eb8a59c07f"),
    "upright_left_mirror_endpoint": (
        "artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json",
        "309002e9c091a15e77c603e0c1c8764d5b3fa7805e54041062920da639389c8c"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    task = (APE / "task.py").read_text()
    foot = (APE / "foot_upright.py").read_text()
    parsed = ast.parse(task)
    eager = tuple(name.name for node in parsed.body
                  if isinstance(node, ast.ImportFrom) and node.level == 1
                  and node.module is None for name in node.names
                  if name.name in EAGER_MODULES)
    if eager != EAGER_MODULES:
        raise RuntimeError(f"ape_walk.task eager registration imports changed: {eager}")
    if ("pose_spec = json.loads((LEFT_ENDPOINT if side == \"left\" else ENDPOINT).read_text())"
            not in foot or foot.count("register_mjlab_task(") != 5):
        raise RuntimeError("foot_upright endpoint read or eager registration changed")
    endpoints = {}
    for name, (relative, expected_sha) in ENDPOINTS.items():
        path = ROOT / relative
        if (relative.split("/")[-1] not in foot or not path.is_file()
                or sha(path) != expected_sha):
            raise RuntimeError(f"Eager endpoint asset missing or changed: {name}")
        endpoints[name] = {"path": relative, "sha256": expected_sha,
                           "source_is_symlink": path.is_symlink(),
                           "packed_as_regular_file": True}
    source = ROOT / "experiments/compact_morphology/mjlab_ape/src"
    pythonpath = f"{source}:{ROOT}"
    code = ("import importlib.util,json,sys;"
            "s=importlib.util.find_spec('ape_walk');"
            "print(json.dumps({'origin':s.origin,'imported':'ape_walk' in sys.modules}))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": pythonpath},
                            text=True, capture_output=True, check=True, timeout=20)
    origin = json.loads(result.stdout)
    if (Path(origin["origin"]).resolve() != (APE / "__init__.py").resolve()
            or origin["imported"]):
        raise RuntimeError("Local pre-import find_spec did not select checkout ape_walk")
    return {"schema": "robot-learning-lab.cove50-locomotion-import-closure.v2",
            "status": "local_static_source_and_asset_gate_passed",
            "task_sha256": sha(APE / "task.py"),
            "foot_upright_sha256": sha(APE / "foot_upright.py"),
            "eager_task_modules": list(eager), "endpoint_assets": endpoints,
            "local_checkout_preimport_origin": str((APE / "__init__.py").resolve()),
            "worker_pythonpath_required": "/root/robot-learning-lab/experiments/compact_morphology/mjlab_ape/src:/root/robot-learning-lab",
            "claim_limit": "Static local audit; installed Linux entrypoint and CUDA import remain untested."}


def main() -> None:
    report = audit()
    ART.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text() != content:
        raise RuntimeError("Existing v2 import audit differs; preserve first evidence")
    OUTPUT.write_text(content)
    print(json.dumps({"output": str(OUTPUT), "endpoint_count": len(report["endpoint_assets"]),
                      "eager_module_count": len(report["eager_task_modules"])}))


if __name__ == "__main__":
    main()
