"""Freeze the first independent whole-teacher screen before any new CUDA visit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-low-level-handoffs/walk-squat-arms-cuda-method.json"
RUNNER = ROOT / "experiments/compact_morphology/cove50_low_level_handoffs/run_broad_teacher_screen.py"
OUT = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-screen-method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError("Method already frozen; do not replace it")
    source = json.loads(SOURCE.read_text())
    slice_cases = [(rail, delay) for rail in (10, 12) for delay in (0, 20)]
    screen = [
        {"seed": 54301 + 4 * i + j, "walk_velocity_m_s": velocity,
         "rail_v": rail, "delay_ms": delay}
        for i, velocity in enumerate((0.20, 0.25, 0.30, 0.35))
        for j, (rail, delay) in enumerate(slice_cases)
    ]
    held = [
        {"seed": 54401 + 4 * i + j, "walk_velocity_m_s": velocity,
         "rail_v": rail, "delay_ms": delay}
        for i, velocity in enumerate((0.20, 0.25, 0.30))
        for j, (rail, delay) in enumerate(slice_cases)
    ]
    method = {
        "schema": "robot-learning-lab.broad-handoff-v4-screen-method.v1",
        "classification": "fresh teacher whole-episode screen; no learner or scored candidate yet",
        "source_v3_method_sha256": sha(SOURCE),
        "new_runner_sha256": sha(RUNNER),
        "exact_body_sha256": source["body"]["sha256"],
        "actors": source["actors"],
        "control_hz": 50, "episode_steps": 1200,
        "command_clock": source["clock"],
        "source_backed_slice": {"walk_velocity_m_s": [0.20, 0.35],
                                "rail_v": [10, 12], "delay_ms": [0, 20],
                                "timing_jitter_s": 0},
        "sentinel": {"seed": 54300, "walk_velocity_m_s": 0.25,
                     "rail_v": 12, "delay_ms": 0, "steps": 1, "unscored": True},
        "screen": screen,
        "reserved_fresh_evaluation": held,
        "accepted_example": "one complete native no-reset episode passing the V3 physical gate; never substitute a failed route or a frame-only fragment",
        "early_stop": "After all 16 independent screen families, stop if zero full physical passes. Report acceptance and costs before further collection.",
        "pilot_compute_cap": {"a40_gpu_hours": 1.0, "listed_rental_usd": 0.59},
        "claim_limit": "Only new source-valid teacher collection and measured throughput; no shared learned policy score, redesign transfer or physical qualification.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(method, indent=2) + "\n")


if __name__ == "__main__":
    main()
