"""Compare prescribed goal offsets across opened route contact failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(group: str, plan_name: str, matrix_name: str, label: str) -> None:
    out = BASE / group
    plan_path = out / plan_name
    matrix_path = out / matrix_name
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != plan["case_count"] or
            matrix["plan_sha256"] != sha(plan_path) or
            [r["diagnostic_y_bias_m"] for r in rows] !=
            [c["diagnostic_y_bias_m"] for c in plan["cases"]]):
        raise ValueError(f"Complete source-matched {group} probe required")
    for row in rows:
        if any(not (ROOT / row[name]).is_file() for name in ("result", "trace", "log")):
            raise ValueError(f"Missing retained native evidence for {row['id']}")
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    if group == "v61-roll":
        ax.bar(range(len(rows)), [r["contact_s"] for r in rows],
               color=["#16846c" if r["pass"] else "#bd5055" for r in rows], width=.65)
        ax.set_xticks(range(len(rows)),
                      [f"{r['diagnostic_y_bias_m']*1000:+.0f} mm" for r in rows])
        ax.axhline(.92, color="#253c61", linestyle="-.",
                   label="V60 raised-roll failure: 0.92 s")
        for index, row in enumerate(rows):
            ax.text(index, row["contact_s"] + .08, f"{row['contact_s']:.2f}",
                    ha="center", fontsize=9)
        figure_name = "prescribed-roll-y-bias-evaluation.png"
    else:
        by_route = {route: [r for r in rows if r["route_variant"] == route]
                    for route in ("pitch-low", "roll-low", "roll-high")}
        colors = {"pitch-low": "#215b9a", "roll-low": "#bc7832", "roll-high": "#7c519e"}
        for route, group_rows in by_route.items():
            if len(group_rows) != 5:
                raise ValueError(f"Missing five {route} offset visits")
            ax.plot([r["diagnostic_y_bias_m"] * 1000 for r in group_rows],
                    [r["contact_s"] for r in group_rows], marker="o", linewidth=2,
                    color=colors[route], label=route)
            for row in group_rows:
                if not row["pass"]:
                    ax.scatter(row["diagnostic_y_bias_m"] * 1000, row["contact_s"],
                               marker="x", color="#bd5055", s=90, zorder=4)
        ax.set_xticks([-4, -3, -2, 0, 2], ["−4", "−3", "−2", "0", "+2"])
        ax.set_xlabel("Additional prescribed body-Y goal offset (mm); ordinary scoped bias is −2 mm")
        figure_name = "prescribed-edge-y-bias-evaluation.png"
    ax.axhline(1, color="#8a3b42", linestyle="--", label="1 s full-gate contact threshold")
    ax.axhline(1.2, color="#c48437", linestyle=":", label="1.2 s margin reference")
    ax.set_ylabel("Continuous fingertip contact (s)")
    ax.set_ylim(0, max(5, max(r["contact_s"] for r in rows) + .6))
    ax.legend(loc="upper right")
    fig.suptitle(f"{label} · {matrix['pass_count']}/{len(rows)} native passes\n"
                 "Rejected historical Direct Frame · prescribed offsets are not a policy",
                 fontsize=13)
    figure_path = out / figure_name
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v61-route-y-bias-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "passing_teachers": [{"id": r["id"],
                              "route_variant": r.get("route_variant", "roll-high"),
                              "additional_y_bias_m": r["diagnostic_y_bias_m"],
                              "contact_s": r["contact_s"]}
                             for r in rows if r["pass"] and r["contact_s"] >= 1.2],
        "failures": [{"id": r["id"], "additional_y_bias_m": r["diagnostic_y_bias_m"],
                      "contact_s": r["contact_s"], "faults": r["faults"]}
                     for r in rows if not r["pass"]],
        "figure": str(figure_path.relative_to(ROOT)), "figure_sha256": sha(figure_path),
        "qualification": "Prescribed opened offsets only; an automatic observed controller must retain neighboring routes and pass separate varied and fresh native gates.",
    }
    (out / "prescribed-route-y-bias-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"group": group, "pass_count": summary["pass_count"],
                      "case_count": summary["case_count"],
                      "teacher_count": len(summary["passing_teachers"]),
                      "failures": summary["failures"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("roll", "edge", "all"), default="all")
    group = parser.parse_args().group
    if group in ("roll", "all"):
        summarize("v61-roll", "prescribed-roll-y-bias-plan.json",
                  "opened-roll-y-bias/matrix.json", "V61 raised-roll goal-offset diagnostic")
    if group in ("edge", "all"):
        summarize("v61-edge", "prescribed-edge-y-bias-plan.json",
                  "opened-edge-y-bias/matrix.json", "V61 edge-view route goal-offset diagnostic")
