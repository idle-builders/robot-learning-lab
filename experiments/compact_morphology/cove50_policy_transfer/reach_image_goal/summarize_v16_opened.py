"""Compare every opened v16 native replay with its exact v14 start."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V16 = BASE / "v16"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = V16 / "opened-native-plan.json"
    matrix_path = V16 / "opened-native/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (matrix["plan_sha256"] != sha(plan_path) or
            not (matrix["complete"] and matrix["case_count"] == 30 or
                 matrix.get("stopped_early") and matrix["case_count"] == 14)):
        raise ValueError("Need the complete gate or its sealed failed early stop")
    old = {}
    for stage in ("development", "fresh-reserved"):
        path = BASE / f"v14/{stage}/matrix.json"
        if sha(path) != plan[f"v14_{'development' if stage == 'development' else 'fresh'}_matrix_sha256"]:
            raise ValueError(f"V14 source matrix changed: {path}")
        old.update({row["seed"]: row for row in json.loads(path.read_text())["cases"]})
    rows = matrix["cases"]
    if len({row["seed"] for row in rows}) != len(rows) or not {row["seed"] for row in rows} <= set(old):
        raise ValueError("V16 and V14 starts are not paired")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axis, (origin, title) in zip(axes, (("prior_v14_development", "Earlier opened 10"),
                                           ("prior_v14_fresh_now_opened", "Prior fresh 20, now opened"))):
        cases = [row for row in rows if row["origin"] == origin]
        for row in cases:
            previous = old[row["seed"]]
            color = "#2b9d78" if row["pass"] else "#bd684d"
            marker = "s" if row["selection_mode"] == "v16_v10_height_locked" else "o"
            axis.scatter(previous["contact_s"], row["contact_s"], color=color,
                         marker=marker, s=55)
            if not previous["pass"] or not row["pass"]:
                axis.annotate(str(row["seed"]), (previous["contact_s"], row["contact_s"]),
                              xytext=(4, -15) if row["seed"] == 10304 else (4, 4),
                              textcoords="offset points", fontsize=9)
        axis.plot([0, 6], [0, 6], color="#8191a0", linewidth=1)
        axis.axhline(1, color="#394b5e", linestyle="--", linewidth=1)
        axis.axvline(1, color="#394b5e", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlim(-.1, 6)
        axis.set_ylim(-.1, 6)
        axis.set_xlabel("V14 contact (s)")
        axis.grid(alpha=.12)
    axes[0].set_ylabel("V16 contact (s)")
    fig.suptitle(f"Matched opened native reach · v14 versus v16 candidate · {len(rows)}/30 observed\n"
                 "Rejected Direct Frame · square = v10 height branch · dashed = 1 s gate", fontsize=14)
    fig.tight_layout()
    fig.savefig(V16 / "opened-native/evaluation.png", dpi=160)
    plt.close(fig)
    failed = [row for row in rows if not row["pass"]]
    summary = {
        "schema": "robot-learning-lab.cove50-v16-opened-native-summary.v1",
        "role": "opened_development_gate_only",
        "plan_sha256": sha(plan_path),
        "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "pass_count": matrix["pass_count"],
        "case_count": matrix["case_count"],
        "planned_case_count": 30,
        "stopped_early": bool(matrix.get("stopped_early")),
        "stop_reason": matrix.get("stop_reason"),
        "by_origin": {origin: {
            "pass": sum(row["pass"] for row in rows if row["origin"] == origin),
            "cases": sum(row["origin"] == origin for row in rows),
        } for origin in ("prior_v14_development", "prior_v14_fresh_now_opened")},
        "by_mode": {mode: {
            "pass": sum(row["pass"] for row in rows if row["selection_mode"] == mode),
            "cases": sum(row["selection_mode"] == mode for row in rows),
        } for mode in sorted({row["selection_mode"] for row in rows})},
        "failed_seeds": [row["seed"] for row in failed],
        "matched_rescues": [row["seed"] for row in rows if row["pass"] and not old[row["seed"]]["pass"]],
        "matched_regressions": [row["seed"] for row in rows if not row["pass"] and old[row["seed"]]["pass"]],
        "fresh_gate_pass": bool(matrix["complete"] and matrix["pass_count"] == 30),
        "qualification": "Historical rejected-body opened replay. Only 30/30 permits a new frozen disjoint candidate score; no physical-camera, grasp or selected-body claim.",
    }
    (V16 / "opened-native/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
