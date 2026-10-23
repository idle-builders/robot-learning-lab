"""Check the stop-reward target against saved V2 native development traces.

This audits labels and phase timing; it does not run physics or score a learner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-low-level-training-20261015/modular-portfolio-breadth-v2/native"
OUT = ROOT / "artifacts/ape-low-level-training-20261015/modular-stop-feedback-prep/reward-alignment-audit.json"


def main() -> None:
    rows = []
    for path in sorted(SOURCE.glob("seed-*-original.json")):
        raw = path.read_bytes()
        report = json.loads(raw)
        stop = report["windows"]["settled"]
        after = report["windows"]["stand_before"]
        schedule = report["schedule_s"]
        if schedule["stop_end"] - schedule["walk_end"] < 1.0:
            raise ValueError("stop phase does not leave a full scored second")
        expected = [r for r in report["timeseries"]
                    if schedule["stop_end"] - 1.0 <= r["t"] < schedule["stop_end"]]
        if len(expected) != stop["frames"] or not 48 <= len(expected) <= 51:
            raise ValueError(f"saved window and raw rows disagree: {path}")
        stop_support = sum(r["feet_both"] for r in expected) / len(expected)
        stop_duty = sum(r["leg_duty"] for r in expected) / len(expected)
        if abs(stop_support - stop["feet_both_fraction"]) > 1e-9 or abs(stop_duty - stop["leg_duty_fraction"]) > 1e-9:
            raise ValueError(f"saved window fractions disagree: {path}")
        stop_window_pass = (stop_support >= 0.95 and stop_duty <= 0.10 and
                            report["stop_tail_xy_m"] <= 0.03)
        after_window_pass = (after["feet_both_fraction"] >= 0.95 and
                             after["leg_duty_fraction"] <= 0.10)
        rows.append({
            "source": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "seed": report["seed"],
            "stop_window_s": [schedule["stop_end"] - 1.0, schedule["stop_end"]],
            "stop_frames": len(expected),
            "stop_support_fraction": stop_support,
            "stop_rated_leg_duty_fraction": stop_duty,
            "stop_tail_m": report["stop_tail_xy_m"],
            "stop_window_pass": stop_window_pass,
            "later_posture_window_pass": after_window_pass,
            "whole_sequence_pass": report["full_continuity_gate_pass"],
        })
    if len(rows) != 16:
        raise ValueError("expected all 16 V2 original native reports")
    if sum(r["whole_sequence_pass"] for r in rows) != 6:
        raise ValueError("V2 whole-sequence source score changed")
    audit = {
        "schema": "robot-learning-lab.modular-stop-reward-alignment.v1",
        "claim_limit": "offline phase/label audit of saved development reports; no learner or new native score",
        "source_evaluator": str(Path("experiments/compact_morphology/cove50_low_level_portfolio/evaluate_portfolio_breadth_v2.py")),
        "source_evaluator_sha256": hashlib.sha256((ROOT / "experiments/compact_morphology/cove50_low_level_portfolio/evaluate_portfolio_breadth_v2.py").read_bytes()).hexdigest(),
        "original_reports": len(rows),
        "stop_window_pass": sum(r["stop_window_pass"] for r in rows),
        "later_posture_window_pass": sum(r["later_posture_window_pass"] for r in rows),
        "whole_sequence_pass": sum(r["whole_sequence_pass"] for r in rows),
        "false_positive_if_later_posture_used": sum(r["later_posture_window_pass"] and not r["stop_window_pass"] for r in rows),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
