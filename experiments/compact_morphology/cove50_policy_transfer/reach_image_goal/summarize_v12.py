"""Seal the historical v12 native reach matrix and make its failures visible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V12 = OUT / "v12"
MODES = {"v9_locked": "#4477AA", "v9_live": "#22A77E", "v10_locked": "#B779CC"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(stage: str, count: int) -> dict:
    path = V12 / stage / "matrix.json"
    data = json.loads(path.read_text())
    if data["case_count"] != count or len(data["cases"]) != count:
        raise ValueError(f"Incomplete {stage} matrix")
    if data["plan_sha256"] != sha(V12 / "plan.json"):
        raise ValueError(f"Changed {stage} plan")
    if len({row["seed"] for row in data["cases"]}) != count:
        raise ValueError(f"Duplicate {stage} seed")
    for row in data["cases"]:
        report = json.loads((ROOT / row["result"]).read_text())
        if (report["seed"] != row["seed"] or
                report["selection_mode"] != row["selection_mode"] or
                report["body_role"] != "rejected_historical_direct_frame_sandbox" or
                report["phases"]["reach"]["continuous_finger_contact_s"] != row["contact_s"]):
            raise ValueError(f"Case identity or outcome changed: {row['seed']}")
    return data


def count_by(rows: list[dict], key: str) -> list[dict]:
    labels = sorted({row[key] for row in rows})
    return [{key: label, "pass_count": sum(row["pass"] for row in rows if row[key] == label),
             "case_count": sum(row[key] == label for row in rows)} for label in labels]


def main() -> None:
    plan = json.loads((V12 / "plan.json").read_text())
    if plan["body_role"] != "rejected_historical_direct_frame_sandbox":
        raise ValueError("Wrong mechanical qualification boundary")
    dev = matrix("development", 8)
    fresh = matrix("fresh-reserved", 20)
    if dev["pass_count"] != 8:
        raise ValueError("Fresh matrix must follow complete opened gate")
    rows = fresh["cases"]
    result = {
        "schema": "robot-learning-lab.cove50-rgb-adaptive-v12-summary.v1",
        "body_role": plan["body_role"],
        "selected_mechanical_design": None,
        "claim": "Historical rejected Direct Frame control sandbox only; no physical camera, grasp or new-body qualification.",
        "plan_sha256": sha(V12 / "plan.json"),
        "evaluator_sha256": plan["sources_sha256"]["evaluator"],
        "v9_actor_sha256": plan["sources_sha256"]["actor"],
        "v10_actor_sha256": plan["sources_sha256"]["alternate_actor"],
        "body_sha256": plan["sources_sha256"]["body"],
        "development_matrix_sha256": sha(V12 / "development/matrix.json"),
        "fresh_matrix_sha256": sha(V12 / "fresh-reserved/matrix.json"),
        "development": {"pass_count": dev["pass_count"], "case_count": dev["case_count"],
                        "by_mode": count_by(dev["cases"], "selection_mode")},
        "fresh": {"pass_count": fresh["pass_count"], "case_count": fresh["case_count"],
                  "by_mode": count_by(rows, "selection_mode"),
                  "by_voltage": count_by(rows, "rail_v"),
                  "by_delay_ms": count_by(rows, "delay_ms"),
                  "contact_s_min": round(min(row["contact_s"] for row in rows), 3),
                  "contact_s_median": round(float(np.median([row["contact_s"] for row in rows])), 3),
                  "entry_rgb_error_mm_max_scoring_only": max(row["entry_goal_error_mm_scoring_only"] for row in rows),
                  "full_two_foot_and_zero_rated_leg_count": sum(
                      row["reach_both_feet_fraction"] == 1.0 and row["reach_rated_leg_fraction"] == 0.0
                      for row in rows),
                  "failures": [{"seed": row["seed"], "mode": row["selection_mode"],
                                "contact_s": row["contact_s"], "faults": row["faults"],
                                "result": row["result"]} for row in rows if not row["pass"]]},
    }
    (V12 / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#ffffff")
    positions = np.arange(len(rows))
    for mode, color in MODES.items():
        part = [i for i, row in enumerate(rows) if row["selection_mode"] == mode]
        if part:
            ax.bar(positions[part], [rows[i]["contact_s"] for i in part], width=.78,
                   color=color, label=mode.replace("_", " "))
    for i, row in enumerate(rows):
        if not row["pass"]:
            ax.text(i, max(.07, row["contact_s"] + .06), "×", ha="center",
                    color="#b3293b", fontsize=17, fontweight="bold")
    ax.axhline(1, color="#b3293b", linestyle="--", linewidth=1.3,
               label="1 s contact gate")
    ax.set_xticks(positions, [str(row["seed"])[-3:] for row in rows], rotation=0)
    ax.set_xlabel("Fresh seed 102xx · 11/12 V, 0/5/10 ms delay, varied toy XY")
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_ylim(0, max(5, max(row["contact_s"] for row in rows) + .6))
    ax.set_title(f"Cove-50 RGB reach v12 · {fresh['pass_count']}/20 fresh native passes · rejected historical body",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.savefig(V12 / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"fresh_pass_count": fresh["pass_count"],
                      "fresh_case_count": fresh["case_count"],
                      "modes": result["fresh"]["by_mode"],
                      "failures": result["fresh"]["failures"]}, indent=2))


if __name__ == "__main__":
    main()
