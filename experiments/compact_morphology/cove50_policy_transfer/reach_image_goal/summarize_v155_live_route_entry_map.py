"""Source-check the opened V155 camera-only route map and render its grid."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
    v118_observable_selector as observations,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v155/opened-live-entry-map-plan.json"
MATRIX = BASE / "v155/native-entry-only/matrix.json"
V149_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
SUMMARY = BASE / "v155/source-checked-entry-map-summary.json"
PARTIAL_SUMMARY = BASE / "v155/partial-source-checked-entry-map-summary.json"
FIGURE = BASE / "v155/live-route-entry-map.png"
PARTIAL_FIGURE = BASE / "v155/partial-live-route-entry-map.png"
COLORS = {"unvisited": "#e4e8e5", "v16_live": "#38a677",
          "v16_locked": "#6686bc", "v21_scoped_y_locked": "#da9a39",
          "v16_specialist_locked": "#ba6a9a", "other": "#b88777",
          "missing_rgb": "#ba4a57"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def draw(plan: dict, rows: dict[str, dict], path: Path) -> None:
    offsets = [-1, -.5, 0, .5, 1]
    names = list(COLORS)
    colors = ListedColormap([COLORS[name] for name in names])
    grid = [[0 for _ in offsets] for _ in offsets]
    labels = [["" for _ in offsets] for _ in offsets]
    for case in plan["cases"]:
        row = rows.get(case["id"])
        if not row:
            continue
        x = offsets.index(round(case["toy_delta_x_m"] * 1000, 1))
        y = offsets.index(round(case["toy_delta_y_m"] * 1000, 1))
        mode = (row["selection_mode"] if row["entry_available"] else "missing_rgb")
        key = mode if mode in COLORS else "other"
        grid[y][x] = names.index(key)
        obs = row["observed_features"]
        short = {"v16_live": "live", "v16_locked": "locked",
                 "v21_scoped_y_locked": "scoped",
                 "v16_specialist_locked": "specialist"}.get(mode, mode)
        labels[y][x] = (f"{short}\n{obs['entry_error_x_mm']:+.1f},"
                        f"{obs['entry_error_y_mm']:+.1f} mm" if obs else
                        "no RGB")
    fig, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(grid, origin="lower", cmap=colors, vmin=0,
                vmax=len(names) - 1)
    axis.set(xticks=range(5), yticks=range(5),
             xticklabels=[f"{v:+g}" for v in offsets],
             yticklabels=[f"{v:+g}" for v in offsets],
             xlabel="Toy X change from opened seed 24591 (mm)",
             ylabel="Toy Y change from opened seed 24591 (mm)",
             title=(f"V155 controlled RGB entry map · {len(rows)}/25 one-seed "
                    "visits · no contact"))
    axis.set_xticks([v - .5 for v in range(1, 5)], minor=True)
    axis.set_yticks([v - .5 for v in range(1, 5)], minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    for y in range(5):
        for x in range(5):
            if labels[y][x]:
                axis.text(x, y, labels[y][x], ha="center", va="center",
                          fontsize=8, color="#16251d")
    fig.tight_layout()
    temporary = path.with_name(path.stem + ".tmp.png")
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-partial", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    prior = json.loads(V149_MATRIX.read_text())
    reserve = json.loads(RESERVE.read_text())
    source = plan["source_sha256"]
    cases = {case["id"]: case for case in plan["cases"]}
    if (source["runner"] != sha(Path(__file__).with_name(
            "run_v155_live_route_entry_map.py")) or
            source["v149_matrix"] != sha(V149_MATRIX) or
            source["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            plan["case_count"] != 25 or
            plan["correlated_anchor_seed_count"] != 1 or
            matrix["plan_sha256"] != sha(PLAN) or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["complete"] != (matrix["case_count"] == 25) or
            (not args.verify_partial and not matrix["complete"]) or
            len({row["id"] for row in matrix["cases"]}) != matrix["case_count"] or
            not {row["id"] for row in matrix["cases"]} <= set(cases) or
            matrix["contains_contact_outcomes"]):
        raise ValueError("V155 frozen camera-only matrix differs")
    anchor = next(row for row in prior["cases"] if row["seed"] == 24591)
    checked = {}
    files = []
    faults = []
    for row in matrix["cases"]:
        case = cases[row["id"]]
        report_path = ROOT / row["result"]
        trace_path = report_path.with_suffix(".npz")
        if (sha(report_path) != row["result_sha256"] or
                sha(trace_path) != row["trace_sha256"]):
            raise ValueError(f"Saved V155 source changed: {row['id']}")
        report = json.loads(report_path.read_text())
        verify_trace(str(trace_path.relative_to(ROOT)), report)
        obs = observations.observed_features(report) if row["entry_available"] else None
        if (report["seed"] != case["seed"] or
                report["evaluator_sha256"] != source["evaluator"] or
                report["body_sha256"] != source["body"] or
                report["selected_pointer_sha256"] != source["pointer"] or
                report["reach_checkpoint_sha256"] != source["actor"] or
                not report["preflight_only"] or
                report["reach_duration_s"] != .02 or
                report["toy_offset_b_m"] != [case["toy_offset_x_m"],
                                              case["toy_offset_y_m"]] or
                report["rail_v"] != case["rail_v"] or
                report["delay_ms"] != case["delay_ms"] or
                report["pitch_offset_rad"] != case["pitch_offset_rad"] or
                report["roll_clearance_rad"] != case["roll_clearance_rad"] or
                report["gaze_head_yaw_rad"] != case["gaze_head_yaw_rad"] or
                obs != row["observed_features"] or
                row["selection_mode"] != report["selection_mode"] or
                row["correlation_family_audit_only"] !=
                "opened-seed-24591"):
            raise ValueError(f"V155 report or RGB entry differs: {row['id']}")
        if row["id"] == "live-map-center-24591":
            parity = score_runner.entry_parity(anchor["observed_features"], obs)
            if parity != row["center_rgb_parity_faults"] or parity:
                raise ValueError("V155 center differs from source-checked V149 RGB")
        if not row["entry_available"]:
            faults.append({"id": row["id"], "reason": row["fault"]})
        files.append({"id": row["id"], "report": row["result"],
                      "trace": str(trace_path.relative_to(ROOT)),
                      "report_sha256": row["result_sha256"],
                      "trace_sha256": row["trace_sha256"]})
        checked[row["id"]] = row
    figure = PARTIAL_FIGURE if args.verify_partial else FIGURE
    draw(plan, checked, figure)
    result = {
        "schema": "robot-learning-lab.cove50-v155-source-checked-entry-map.v1",
        "role": "opened_one_seed_camera_only_route_map_not_contact_or_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "v149_matrix": sha(V149_MATRIX),
                          "reserve": sha(RESERVE),
                          "checker": sha(Path(__file__)),
                          "figure": sha(figure)},
        "source_checked_entry_count": len(checked),
        "planned_count": 25, "complete": len(checked) == 25,
        "correlated_anchor_seed_count": 1,
        "contains_contact_outcomes": False,
        "center_rgb_parity_fault_count": 0,
        "entry_available_count": sum(row["entry_available"] for row in checked.values()),
        "observed_route_counts": dict(sorted(Counter(
            row["selection_mode"] for row in checked.values()).items())),
        "faults": faults,
        "source_files": files,
        "interpretation": (
            "This grid maps controlled RGB route exposure on one already "
            "opened simulator seed. It has no physical contact outcomes and "
            "cannot supply independent held repair support or qualify a "
            "policy. Use it only to choose a newly frozen varied-seed "
            "camera-entry search."),
    }
    write_json(PARTIAL_SUMMARY if args.verify_partial else SUMMARY, result)
    print(json.dumps({key: result[key] for key in (
        "source_checked_entry_count", "complete", "entry_available_count",
        "observed_route_counts")}))


if __name__ == "__main__":
    main()
