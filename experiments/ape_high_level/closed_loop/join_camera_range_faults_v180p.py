"""Join saved camera blockage and forward-range occlusion by scene identity.

This is a diagnostic fusion of previously simulated actor cues. It does not
create a new live owner turn or make an observation/response reservation.
"""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import compare_standoff_range_layouts_v165p as v165p
import stress_camera_depth_packets_v179p as v179p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-camera-range-fault-join"
CAMERA_CONDITION = "moderate_contiguous_40pct"
RANGE_STRESS = "forward_pair_occluded"
RANGE_LAYOUT = "four_front_side"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    camera_file = v179p.OUT / "report.json"
    range_file = v165p.OUT / "report.json"
    camera = json.loads(camera_file.read_text())
    ranges = json.loads(range_file.read_text())
    by_scene = {row["source_case_id"]: row for row in ranges["rows"]
                if row["stress"] == RANGE_STRESS and
                row["layout"] == RANGE_LAYOUT}
    selected = [row for row in camera["rows"]
                if row["condition"] == CAMERA_CONDITION]
    pairs = {}
    for row in selected:
        pairs.setdefault((row["source_case_id"], row["draw"]),
                         {})[row["camera"]] = row
    rows = []
    for (case, draw), views in sorted(pairs.items()):
        if set(views) != set(v179p.v171p.CAMERAS) or case not in by_scene:
            raise AssertionError("Unmatched camera/range source case")
        range_row = by_scene[case]
        first = [time for time in (
            range_row["first_front_left_cue_s"],
            *(v["first_cue_s"] for v in views.values()))
            if time is not None]
        first_any = min(first) if first else None
        breach = range_row["evaluator_first_below_5cm_s"]
        if breach != next(iter(views.values()))[
                "evaluator_first_below_5cm_s"]:
            raise AssertionError("Evaluator breach label differs")
        is_hazard = range_row["class"] == "hazard"
        rows.append({
            "source_case_id": case,
            "class": range_row["class"],
            "camera_fault_draw": draw,
            "range_first_cue_s": range_row["first_front_left_cue_s"],
            "head_first_cue_s": views["lower_rgb"]["first_cue_s"],
            "chest_first_cue_s": views["chest_low70_x45"]["first_cue_s"],
            "first_any_source_cue_s": first_any,
            "evaluator_first_below_5cm_s": breach,
            "timely_hazard_warning": (is_hazard and first_any is not None
                                      and breach is not None and
                                      first_any <= breach),
            "false_control_warning": (not is_hazard and
                                      first_any is not None),
        })
    hazards = [row for row in rows if row["class"] == "hazard"]
    controls = [row for row in rows if row["class"] == "control"]
    result = {
        "schema": "robot-learning-lab.ape-camera-range-fault-join.v1",
        "source_sha256": sha(SOURCE),
        "camera_report_sha256": sha(camera_file),
        "range_report_sha256": sha(range_file),
        "camera_condition": CAMERA_CONDITION,
        "range_stress": RANGE_STRESS,
        "range_layout": RANGE_LAYOUT,
        "scope": "Same eight saved hazard and eight control scene identities: "
                 "five independently sampled camera perturbation draws per "
                 "case joined to one saved four-view range forward-pair "
                 "occlusion replay. Earliest actor-visible cue wins. This is "
                 "an offline fusion under simultaneous synthetic faults, "
                 "not 40 independent scenes, calibrated cross-modal "
                 "occlusion, an invitation gate or a full owner response.",
        "summary": {
            "hazard_trials": len(hazards),
            "range_timely_warnings": sum(
                by_scene[row["source_case_id"]]["timely_hazard_warning"]
                for row in hazards),
            "camera_either_timely_warnings": sum(
                any(row[key] is not None and
                    row[key] <= row["evaluator_first_below_5cm_s"]
                    for key in ("head_first_cue_s", "chest_first_cue_s"))
                for row in hazards),
            "fused_timely_warnings": sum(
                row["timely_hazard_warning"] for row in hazards),
            "control_trials": len(controls),
            "fused_false_control_warnings": sum(
                row["false_control_warning"] for row in controls),
        },
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["summary"])


if __name__ == "__main__":
    main()
