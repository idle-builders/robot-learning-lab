"""V35 A40-r1 installed-runtime gate.

V31d's original probe is immutable and intentionally RTX 4090-specific. This
prospective revision checks the same complete package/import/source closure,
then compares every non-hardware field with V31d while recording the A40
identity as an explicit hardware shift. No native trajectory is run here.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v31d as v31d


ROOT = Path(__file__).resolve().parents[3]
GPU = "NVIDIA A40"
OLD_GPU = "NVIDIA GeForce RTX 4090"
EXPECTED = v31d.EXPECTED
MODULES = v31d.MODULES
BAM_COMMIT = v31d.BAM_COMMIT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closure(plan: dict) -> dict:
    """Reproduce V31d's 227-distribution inspection with a pinned A40."""
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux":
        raise RuntimeError("V35 A40-r1 requires Linux Python 3.12")
    versions = {name: metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"V35 native package versions differ: {versions}")
    pip_check = v31d.run_command("check").strip()
    inspect = json.loads(v31d.run_command("inspect"))
    installed = []
    bam_direct_url = None
    for item in inspect["installed"]:
        meta = item["metadata"]
        name = meta["name"].lower().replace("_", "-")
        if name == "better-actuator-models":
            bam_direct_url = item.get("direct_url")
        installed.append({"name": name, "version": meta["version"],
                          "location": item.get("metadata_location"),
                          "direct_url": item.get("direct_url")})
    installed.sort(key=lambda item: (item["name"], item["version"],
                                     item["location"] or ""))
    duplicates = sorted({item["name"] for item in installed
                         if sum(p["name"] == item["name"] for p in installed) > 1})
    if any(name in duplicates for name in EXPECTED):
        raise RuntimeError(f"Duplicate critical distributions: {duplicates}")
    if (not isinstance(bam_direct_url, dict)
            or bam_direct_url.get("vcs_info", {}).get("commit_id") != BAM_COMMIT
            or bam_direct_url.get("dir_info", {}).get("editable")):
        raise RuntimeError("BAM installed source is not the exact noneditable commit")
    modules = {}
    for name in MODULES:
        module = importlib.import_module(name)
        modules[name] = str(Path(module.__file__).resolve())
    import ape_walk.task as ape_task
    from ape_walk.physics import frame50 as frame50_source
    for key, module in (("task", ape_task), ("frame50", frame50_source)):
        origin = Path(module.__file__).resolve()
        expected_origin = (ROOT / plan["source_paths"][key]).resolve()
        if origin != expected_origin:
            raise RuntimeError(f"V35 {key} loaded from wrong source path: {origin}")
        if sha(origin) != plan["source_sha256"][key]:
            raise RuntimeError(f"V35 {key} imported source differs from frozen plan")
        modules["ape_walk.task" if key == "task" else "ape_walk.physics.frame50"] = str(origin)
    import bam.mjlab as bam_mjlab
    import mjlab_microduck.tasks.mdp as upstream_mdp
    upstream_sha = sha(Path(upstream_mdp.__file__))
    bam_sha = sha(Path(bam_mjlab.__file__))
    if upstream_sha != plan["source_sha256"]["upstream_mdp"]:
        raise RuntimeError("Installed Microduck MDP source differs from frozen reference")
    if bam_sha != plan["source_sha256"]["bam_mjlab_reference"]:
        raise RuntimeError("Installed BAM MJLab source differs from frozen reference")
    import torch
    import warp as wp
    if torch.__version__ != EXPECTED["torch"] or torch.version.cuda != "12.8":
        raise RuntimeError("Torch import differs from the CUDA image pin")
    for name in ("torch", "triton"):
        if Path(modules[name]).is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"{name} was overlaid inside the V31d virtual environment")
    if not hasattr(wp, "context") or not hasattr(wp.context, "runtime"):
        raise RuntimeError("MJLab 1.3.0 requires Warp 1.12's context.runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one working CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         text=True, capture_output=True, check=True, timeout=10).stdout.strip().splitlines()
    if gpu != GPU or smi != [GPU]:
        raise RuntimeError(f"V35 A40-r1 GPU identity differs: Torch={gpu}, nvidia-smi={smi}")
    current = {
        "schema": "robot-learning-lab.cove50-v31d-installed-runtime.v1",
        "python": sys.version.split()[0], "executable": sys.executable,
        "expected_versions": EXPECTED, "installed": installed,
        "import_origins": modules, "pip_check": pip_check,
        "torch_cuda": torch.version.cuda, "gpu": gpu,
        "warp_context_runtime_present": True,
        "pollen_lock_sha256": plan["source_sha256"]["pollen_lock"],
        "installed_upstream_mdp_sha256": upstream_sha,
        "installed_bam_mjlab_sha256": bam_sha,
        "bam_direct_url": bam_direct_url,
        "pip_freeze": v31d.run_command("freeze", "--all").splitlines(),
    }
    current["content_sha256"] = hashlib.sha256(
        json.dumps(current, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return current


def compare_except_hardware(current: dict, prior: dict, expected_prior_sha: str) -> dict:
    """Require all V31d closure fields, including 227 packages, except GPU."""
    if (prior.get("content_sha256") != expected_prior_sha
            or len(prior.get("installed", [])) != 227
            or prior.get("gpu") != OLD_GPU):
        raise RuntimeError("Saved V31d baseline is not the reviewed 227-package RTX closure")
    if current.get("gpu") != GPU or len(current.get("installed", [])) != 227:
        raise RuntimeError("A40 identity or 227-package cardinality differs")
    for key in ("content_sha256",):
        clone = dict(current)
        clone.pop(key, None)
        if hashlib.sha256(json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != current[key]:
            raise RuntimeError("Current closure content hash is invalid")
    differing = sorted(key for key in set(current) | set(prior)
                       if key not in ("gpu", "content_sha256")
                       and current.get(key) != prior.get(key))
    if differing:
        raise RuntimeError(f"A40 installed package/import/source closure differs from V31d: {differing}")
    return {
        "baseline_gpu": OLD_GPU, "current_gpu": GPU,
        "hardware_shift": "A40 CUDA evaluation; no RTX4090 pooling",
        "matched_fields": sorted(set(prior) - {"gpu", "content_sha256"}),
        "matched_installed_distributions": 227,
        "baseline_closure_content_sha256": expected_prior_sha,
        "current_closure_content_sha256": current["content_sha256"],
    }


def static_smoke() -> dict:
    """Catch a copied RTX-only runtime call before provisioning a pod."""
    package = Path(__file__).parent
    active = {
        "runtime": Path(__file__),
        "checkout": package / "checkout_import_gate_v35_a40_r1.py",
        "runner": package / "run_seated_bc_eval_v35_a40_r1.py",
    }
    for name, path in active.items():
        source = path.read_text()
        parsed = ast.parse(source)
        if any(isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name)
               and node.func.value.id == "v31d"
               and node.func.attr == "installed_closure"
               for node in ast.walk(parsed)):
            raise RuntimeError(f"Active A40 {name} still calls RTX-only V31d runtime enforcement")
        if name != "runtime" and "preflight_seated_runtime_v35_a40_r1" not in source:
            raise RuntimeError(f"Active A40 {name} does not use revised runtime gate")
    prior = json.loads((ROOT / "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/installed-closure.json").read_text())
    synthetic = dict(prior, gpu=GPU)
    synthetic.pop("content_sha256")
    synthetic["content_sha256"] = hashlib.sha256(
        json.dumps(synthetic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    comparison = compare_except_hardware(synthetic, prior, prior["content_sha256"])
    altered = json.loads(json.dumps(synthetic))
    altered["installed"][0]["version"] = "unexpected"
    altered.pop("content_sha256")
    altered["content_sha256"] = hashlib.sha256(
        json.dumps(altered, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    try:
        compare_except_hardware(altered, prior, prior["content_sha256"])
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Package drift was accepted by A40 prelaunch smoke")
    return comparison


if __name__ == "__main__":
    print(json.dumps(static_smoke(), sort_keys=True))
