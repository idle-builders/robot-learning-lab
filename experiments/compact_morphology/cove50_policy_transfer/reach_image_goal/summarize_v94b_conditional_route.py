"""Plot measured V94b natural and prescribed routes without changing its frozen gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v94b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "conditional-route-variance-plan.json"
    matrix_path = OUT / "native-conditional-route-variance/matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["condition_count"] != 6 or
            matrix["case_count"] != 10 or matrix["plan_sha256"] != sha(plan_path)):
        raise ValueError("Need complete source-matched V94b conditional matrix")
    pairs = matrix["pairs"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), layout="constrained")
    for index, pair in enumerate(pairs):
        natural_scoped = pair["baseline_route"] == "v21_scoped_y_locked"
        color = "#b94847" if not pair["baseline_pass"] else "#43856b"
        x0 = index - (.15 if natural_scoped else 0)
        ax.scatter(x0, pair["baseline_contact_s"], color=color, marker="o", s=95, zorder=3)
        if natural_scoped:
            ax.plot([index - .15, index + .15],
                    [pair["baseline_contact_s"], pair["forced_contact_s"]],
                    color="#7590aa", linewidth=2, zorder=2)
            ax.scatter(index + .15, pair["forced_contact_s"], color="#345c88",
                       marker="s", s=95, zorder=3)
    ax.axhline(1.0, color="#aa4848", linestyle="--", linewidth=1,
               label="native contact gate: 1.0 s")
    ax.axhline(1.2, color="#b28e36", linestyle=":", linewidth=1,
               label="development margin: 1.2 s")
    ax.scatter([], [], color="#b94847", marker="o", label="natural scoped: fail")
    ax.scatter([], [], color="#43856b", marker="o", label="natural route: pass")
    ax.scatter([], [], color="#345c88", marker="s", label="prescribed specialist: pass")
    ax.set(xticks=range(len(pairs)),
           xticklabels=[pair["condition"].replace("-", "\n") for pair in pairs],
           ylabel="continuous fingertip contact (s)",
           title="V94b: two rescues, two preserved passes; frozen three-rescue gate failed",
           ylim=(0, 4.5))
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    figure_path = OUT / "conditional-route-variance-evaluation.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v94b-conditional-route-summary.v1",
        "role": matrix["role"], "body_role": matrix["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "condition_count": matrix["condition_count"],
        "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
        "scoped_pair_count": matrix["scoped_pair_count"],
        "natural_specialist_count": matrix["natural_specialist_count"],
        "paired_rescue_count": matrix["paired_rescue_count"],
        "diagnostic_gate_met": matrix["diagnostic_gate_met"],
        "figure": str(figure_path.relative_to(ROOT)),
        "figure_sha256": sha(figure_path),
        "qualification": ("Prescribed teacher comparison on the rejected historical "
                          "body. Its frozen diagnostic gate failed; no automatic policy "
                          "or selected mechanical body is qualified."),
    }
    (OUT / "conditional-route-variance-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("condition_count", "case_count", "pass_count",
                       "paired_rescue_count", "diagnostic_gate_met")}, indent=2))


if __name__ == "__main__":
    main()
