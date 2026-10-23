"""V35 worker entry: prove checkout origin before the first ape_walk import.

This is a transport/runtime gate. The reviewed V35 evaluator and scorer remain
byte-identical to the scientific plan.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[3]
WORKER_ROOT = Path("/root/robot-learning-lab")
APE = ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk"
PYTHONPATH = f"{APE.parent}:{ROOT}"
MANIFEST = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/source-manifest.json"
PLAN = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/plan.json"
RUNTIME = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/runtime-attestation.json"
CLOSURE_REPORT = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/installed-closure-v35.json"
EVALUATOR = ROOT / "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_v35_a40_r2.py"
EAGER = ("reach", "grasp", "grasp_tracking", "place_tracking", "place_stages", "foot_upright")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preimport() -> dict:
    if ROOT != WORKER_ROOT or os.environ.get("PYTHONPATH") != PYTHONPATH:
        raise RuntimeError(f"V35 requires checkout at {WORKER_ROOT} and exact PYTHONPATH={PYTHONPATH}")
    if any(name == "ape_walk" or name.startswith("ape_walk.") for name in sys.modules):
        raise RuntimeError("ape_walk imported before V35 checkout assertion")
    spec = importlib.util.find_spec("ape_walk")
    if (spec is None or spec.origin is None or not spec.submodule_search_locations
            or Path(spec.origin).resolve() != (APE / "__init__.py").resolve()
            or [Path(p).resolve() for p in spec.submodule_search_locations] != [APE.resolve()]):
        raise RuntimeError("V35 pre-import ape_walk origin is not the checkout")
    return {"pythonpath": PYTHONPATH, "origin": str(APE / "__init__.py")}


def attest() -> dict:
    origin = preimport()
    manifest = json.loads(MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    if (manifest["plan_sha256"] != sha(PLAN)
            or manifest["v31d_installed_closure_content_sha256"]
            != plan["v31d_installed_closure_content_sha256"]
            or manifest["checkout_gate_path"] != str(Path(__file__).resolve().relative_to(ROOT))
            or manifest["checkout_gate_sha256"] != sha(Path(__file__))
            or manifest["evaluator_path"] != plan["source_paths"]["evaluator"]
            or manifest["evaluator_sha256"] != plan["source_sha256"]["evaluator"]
            or EVALUATOR != ROOT / manifest["evaluator_path"]
            or not EVALUATOR.is_file()
            or sha(EVALUATOR) != manifest["evaluator_sha256"]):
        raise RuntimeError("V35 source manifest is not bound to the plan and V31d closure")
    task = importlib.import_module("ape_walk.task")
    expected = manifest["eager_source_sha256"]
    for name in ("task", *EAGER):
        module = task if name == "task" else importlib.import_module(f"ape_walk.{name}")
        path = Path(module.__file__).resolve()
        if path != (APE / f"{name}.py").resolve() or sha(path) != expected[name]:
            raise RuntimeError(f"V35 eager ape_walk.{name} origin or bytes differ")
    for name, record in manifest["eager_assets"].items():
        path = ROOT / record["path"]
        if not path.is_file() or path.is_symlink() or sha(path) != record["sha256"]:
            raise RuntimeError(f"V35 eager task asset differs: {name}")
    return {"preimport": origin, "eager_source_sha256": expected,
            "eager_assets": manifest["eager_assets"]}


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "transport-self-check":
        # Local only: run the actual checkout evaluator dispatch from an extracted
        # worker bundle through argparse, without opening a CUDA trajectory.
        plan = json.loads(PLAN.read_text())
        manifest = json.loads(MANIFEST.read_text())
        if (manifest["plan_sha256"] != sha(PLAN)
                or manifest["evaluator_path"] != plan["source_paths"]["evaluator"]
                or manifest["evaluator_sha256"] != plan["source_sha256"]["evaluator"]
                or EVALUATOR != ROOT / manifest["evaluator_path"]
                or not EVALUATOR.is_file()
                or sha(EVALUATOR) != manifest["evaluator_sha256"]):
            raise RuntimeError("V35 extracted evaluator dispatch is missing or unbound")
        sys.argv = [str(EVALUATOR), "--help"]
        try:
            runpy.run_path(str(EVALUATOR), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        else:
            raise RuntimeError("V35 evaluator did not parse --help")
        print(json.dumps({"status": "extracted_checkout_dispatch_passed",
                          "evaluator_path": manifest["evaluator_path"],
                          "evaluator_sha256": sha(EVALUATOR)}, sort_keys=True))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "attest":
        print(json.dumps(attest(), sort_keys=True))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "runtime":
        sources = attest()
        from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v35_a40_r2 as a40_runtime
        from experiments.compact_morphology.cove50_stand_recovery import prepare_seated_bc_eval_v35_a40_r2 as prep

        prior = json.loads((prep.V31D / "installed-closure.json").read_text())
        current = a40_runtime.closure(prep.V31D_PLAN)
        comparison = a40_runtime.compare_except_hardware(
            current, prior, json.loads(PLAN.read_text())[
                "v31d_installed_closure_content_sha256"])
        if RUNTIME.exists() or CLOSURE_REPORT.exists():
            raise FileExistsError("V35 runtime evidence already exists; preserve the first worker observation")
        CLOSURE_REPORT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        RUNTIME.write_text(json.dumps({
            "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r2.runtime-attestation",
            "plan_sha256": sha(PLAN), "closure_content_sha256": current["content_sha256"],
            "closure_file_sha256": sha(CLOSURE_REPORT),
            "installed_count": len(current["installed"]), "source": sources,
            "hardware_shift_comparison": comparison,
            "status": "installed_source_attested_before_v35_cuda",
        }, indent=2, sort_keys=True) + "\n")
        print(RUNTIME)
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "runner":
        attest()
        runtime = json.loads(RUNTIME.read_text())
        current = json.loads(CLOSURE_REPORT.read_text())
        prior = json.loads((ROOT / "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/installed-closure.json").read_text())
        from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v35_a40_r2 as a40_runtime
        comparison = a40_runtime.compare_except_hardware(
            current, prior, json.loads(PLAN.read_text())[
                "v31d_installed_closure_content_sha256"])
        if (runtime["plan_sha256"] != sha(PLAN)
                or runtime["closure_content_sha256"] != current["content_sha256"]
                or runtime["closure_file_sha256"] != sha(CLOSURE_REPORT)
                or runtime["hardware_shift_comparison"] != comparison
                or runtime["status"] != "installed_source_attested_before_v35_cuda"):
            raise RuntimeError("V35 reviewed runtime attestation is missing or differs")
        from experiments.compact_morphology.cove50_stand_recovery import run_seated_bc_eval_v35_a40_r2 as runner

        runner.EVALUATOR = Path(__file__).resolve()
        sys.argv = [str(runner.__file__), *sys.argv[2:]]
        runner.main()
        return
    # The frozen runner passes the checkpoint as the first evaluator argument.
    attest()
    sys.argv = [str(EVALUATOR), *sys.argv[1:]]
    runpy.run_path(str(EVALUATOR), run_name="__main__")


if __name__ == "__main__":
    main()
