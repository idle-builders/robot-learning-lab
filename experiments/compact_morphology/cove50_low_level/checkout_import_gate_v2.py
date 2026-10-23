"""Require the exact extracted ape checkout before any task entrypoint import.

The installed mjlab-ape wheel may infer /root as its repository root and
eagerly load unrelated registered tasks. This worker gate runs before that
first import, including in each evaluator subprocess.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[3]
WORKER_ROOT = Path("/root/robot-learning-lab")
APE_SOURCE = ROOT / "experiments/compact_morphology/mjlab_ape/src"
EXPECTED_PYTHONPATH = f"{APE_SOURCE}:{ROOT}"
EXPECTED_INIT = APE_SOURCE / "ape_walk/__init__.py"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
PLAN = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014/cuda-launch-v2/plan.json"
EAGER_MODULES = ("reach", "grasp", "grasp_tracking", "place_tracking",
                 "place_stages", "foot_upright")


def require_checkout_origin() -> dict:
    if ROOT != WORKER_ROOT:
        raise RuntimeError(f"Exact worker extraction root differs: {ROOT}")
    if os.environ.get("PYTHONPATH") != EXPECTED_PYTHONPATH:
        raise RuntimeError(f"Set exact checkout-first PYTHONPATH={EXPECTED_PYTHONPATH}")
    if any(name == "ape_walk" or name.startswith("ape_walk.") for name in sys.modules):
        raise RuntimeError("ape_walk was imported before checkout-origin assertion")
    spec = importlib.util.find_spec("ape_walk")
    if (spec is None or spec.origin is None or not spec.submodule_search_locations
            or Path(spec.origin).resolve() != EXPECTED_INIT.resolve()
            or [Path(path).resolve() for path in spec.submodule_search_locations]
            != [EXPECTED_INIT.parent.resolve()]):
        raise RuntimeError("Pre-import ape_walk find_spec points away from checkout source")
    return {"pythonpath": EXPECTED_PYTHONPATH,
            "ape_walk_preimport_origin": str(EXPECTED_INIT)}


def attest_eager_modules(plan: dict) -> dict:
    """Run task registration once, then reject any wheel/source mixture."""
    import importlib

    importlib.import_module("ape_walk.task")
    result = {}
    for name in EAGER_MODULES:
        module = importlib.import_module(f"ape_walk.{name}")
        path = Path(module.__file__).resolve()
        expected = (ROOT / plan["source_paths"][name]).resolve()
        content_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if path != expected or content_sha != plan["source_sha256"][name]:
            raise RuntimeError(f"Imported eager ape_walk.{name} origin or bytes differ")
        result[name] = {"origin": str(path), "sha256": content_sha}
    return result


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "evaluate":
        raise SystemExit("Usage: checkout_import_gate_v2.py evaluate <evaluator arguments>")
    require_checkout_origin()
    attest_eager_modules(json.loads(PLAN.read_text()))
    sys.argv = [str(EVALUATOR), *sys.argv[2:]]
    runpy.run_path(str(EVALUATOR), run_name="__main__")


if __name__ == "__main__":
    main()
