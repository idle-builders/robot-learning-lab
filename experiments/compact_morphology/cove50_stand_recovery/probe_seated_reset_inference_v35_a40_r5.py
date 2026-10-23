"""Local negative control for the A40-r4 MJLab manual-reset failure.

This checks the installed MJLab/PyTorch behavior on CPU and inspects the exact
prospective evaluator. It does not run a native task or assign a skill score.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import torch
from mjlab.utils.buffers.delay_buffer import DelayBuffer
import mjlab.utils.buffers.delay_buffer as delay_module


ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = ROOT / "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_v35_a40_r5.py"
OUT = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r5/reset-context-static-control.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluator_reset_is_inferred() -> bool:
    tree = ast.parse(EVALUATOR.read_text(), filename=str(EVALUATOR))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if ast.unparse(node.func) != "env.unwrapped.reset":
            continue
        enclosing = node
        while enclosing in parents:
            enclosing = parents[enclosing]
            if isinstance(enclosing, ast.With) and any(
                ast.unparse(item.context_expr) == "torch.inference_mode()"
                for item in enclosing.items
            ):
                matches.append(True)
                break
        else:
            matches.append(False)
    return matches == [True]


def main() -> None:
    with torch.inference_mode():
        buffer = DelayBuffer(min_lag=0, max_lag=1, batch_size=1, device="cpu")
        buffer.append(torch.zeros((1, 2)))
    outside_error = None
    try:
        buffer.reset(torch.tensor([0], dtype=torch.int64))
    except RuntimeError as exc:
        outside_error = str(exc)
    if outside_error is None or "Inplace update to inference tensor outside InferenceMode" not in outside_error:
        raise RuntimeError("Negative control did not reproduce the r4 PyTorch reset failure")
    with torch.inference_mode():
        buffer.reset(torch.tensor([0], dtype=torch.int64))
    if not evaluator_reset_is_inferred():
        raise RuntimeError("Prospective evaluator manual reset is not enclosed in inference mode")
    report = {
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r5.reset-context-static-control",
        "status": "negative_outside_failed_positive_inside_passed_no_native_task",
        "torch_version": torch.__version__,
        "mjlab_delay_buffer_source_sha256": sha(Path(delay_module.__file__)),
        "evaluator_sha256": sha(EVALUATOR),
        "negative_error": outside_error,
        "manual_reset_ast_in_inference_mode": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and json.loads(OUT.read_text()) != report:
        raise FileExistsError("Reset static control changed; supersede packet rather than replace")
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
