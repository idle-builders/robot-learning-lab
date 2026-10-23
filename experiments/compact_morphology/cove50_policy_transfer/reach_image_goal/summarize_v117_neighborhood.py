"""Compare a prescribed local teacher across five camera-near routes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v117"
PLAN = OUT / "neighborhood-transfer-plan.json"
MATRIX = OUT / "native-neighborhood-transfer/matrix.json"
INSPECTOR_MATRIX = OUT / "neighborhood-inspector-matrix.json"
FIGURE = OUT / "neighborhood-contact.png"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
V113_MATRIX = BASE / "v113/native-varied-paired-contact-data/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    plan = json.loads(PLAN.read_text())
    if (matrix["plan_sha256"] != sha(PLAN) or
            plan["v113_training_plan_sha256"] != sha(V113_PLAN)):
        raise ValueError("V117 neighborhood sources differ from frozen plans")
    source = json.loads(V113_MATRIX.read_text())
    if source["plan_sha256"] != sha(V113_PLAN):
        raise ValueError("V113 paired source matrix does not match its plan")
    paired = {row["seed"]: row for row in source["cases"]}
    planned = {case["id"]: case for case in plan["cases"]}
    rows = []
    for row in matrix["cases"]:
        case = planned.get(row["id"])
        if case is None:
            raise ValueError(f"Unplanned V117 case: {row['id']}")
        reference = paired.get(row["seed"])
        rows.append({**row,
                     "camera_distance_from_teacher_mm": case["camera_distance_from_19036_mm"],
                     "target_shift_mm": -3,
                     "unchanged_contact_s": None if not reference else reference["baseline_contact_s"],
                     "minus2_contact_s": None if not reference else reference["shifted_contact_s"],
                     "unchanged_pass": None if not reference else reference["baseline_pass"],
                     "minus2_pass": None if not reference else reference["shifted_pass"],
                     "paired_reference_available": reference is not None})
    ordered = [next((row for row in rows if row["id"] == case["id"]), None)
               for case in plan["cases"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
    ax.axvline(1.2, color="#b98224", linewidth=1.3, linestyle="--",
               label="1.2 s development margin")
    for y, row in enumerate(ordered):
        if row is None:
            continue
        for value, marker, color in (
                (row["unchanged_contact_s"], "s", "#65777d"),
                (row["minus2_contact_s"], "o", "#426b8c"),
                (row["contact_s"], "^", "#087e70" if row["pass"] and
                 row["margin_1p2_s_met"] else "#b74c5a")):
            if value is not None:
                ax.scatter(value, y, marker=marker, s=70, color=color,
                           edgecolor="white", linewidth=.7, zorder=3)
        if row["unchanged_contact_s"] is not None:
            ax.plot([row["unchanged_contact_s"], row["contact_s"]],
                    [y, y], color="#a3adb2", linewidth=1.3, zorder=1)
    ax.set(yticks=range(len(ordered)),
           yticklabels=[f"{case['seed']}  {case['route_variant']}"
                        for case in plan["cases"]],
           xlabel="Longest continuous fingertip contact (s)",
           xlim=(-.1, 6.4), ylim=(-.6, len(ordered)-.4))
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    fig.suptitle(f"Observed-camera neighborhood · −3 mm teacher · "
                 f"{matrix['case_count']}/5 native visits", fontsize=12)
    for marker, color, label in (("s", "#65777d", "V113 unchanged"),
                                 ("o", "#426b8c", "V113 −2 mm"),
                                 ("^", "#087e70", "V117 −3 mm with margin"),
                                 ("^", "#b74c5a", "V117 −3 mm failure/thin")):
        ax.scatter([], [], marker=marker, color=color, label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=4,
              frameon=False, fontsize=8)
    temp = FIGURE.with_name(FIGURE.stem + ".tmp.png")
    fig.savefig(temp, dpi=145)
    plt.close(fig)
    os.replace(temp, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v117-neighborhood-inspector.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "source_matrix_sha256": sha(MATRIX),
        "v113_reference_matrix_sha256": sha(V113_MATRIX),
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "case_count": matrix["case_count"], "planned_count": 5,
        "complete": matrix["complete"],
        "pass_count": matrix["pass_count"],
        "margin_1p2_s_count": matrix["margin_1p2_s_count"],
        "paired_reference_count": sum(row["paired_reference_available"] for row in rows),
        "cases": rows,
        "qualification": plan["qualification"],
    }
    temp = INSPECTOR_MATRIX.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temp, INSPECTOR_MATRIX)
    print(f"V117 {matrix['case_count']}/5 native visits, "
          f"{summary['paired_reference_count']} source-matched pairs", flush=True)


if __name__ == "__main__":
    main()
