"""Audit the 19-room timed image-worker and moving-prop comparison."""

from __future__ import annotations

import json
import math

import numpy as np

from real_capture_dynamic_scene import TexturedCaptureScene, TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import audit_cross_room_inference_response as prior_audit
import run_item_inference_response_19 as study


AUDIT = ART / "item_inference_response_19_audit.json"


def expanded_case(case: dict) -> dict:
    summary = json.loads((ROOT / case["teaching_summary"]).read_text())
    capture = json.loads((ROOT / summary["source_capture"]).read_text())
    if summary["scenario_placement_screen"] != case["screen"]:
        raise ValueError("Source teaching and frozen placement screens differ")
    return {**case, "owner_trace": summary["teaching"]["owner_approach_trace"],
        "source_screen": summary["scenario_placement_screen"],
        "source_capture": summary["source_capture"],
        "source_mesh": capture["source_mesh"]}


def main() -> None:
    frozen = json.loads(study.PRECOMMIT.read_text())
    for key, digest in (("lane_screen", "lane_screen_sha256"),
                        ("cohort", "cohort_sha256"),
                        ("prior_runner", "prior_runner_sha256")):
        if sha(ROOT / frozen[key]) != frozen[digest]:
            raise ValueError("Frozen source changed: " + key)
    if len(frozen["cases"]) != 19 or len({c["capture_group"] for c in
            frozen["cases"]}) != 19:
        raise ValueError("Expected 19 independent supported captures")
    lane = json.loads(study.LANES.read_text())
    if lane["passed"] != 19 or len(lane["rows"]) != 20:
        raise ValueError("Lane-screen denominator changed")
    for case in frozen["cases"]:
        for path, digest in (("teaching_summary", "teaching_summary_sha256"),
                             ("screen", "screen_sha256")):
            if sha(ROOT / case[path]) != case[digest]:
                raise ValueError("Case input changed")
        selected = next(x for x in lane["rows"] if
                        x["scene_key"] == case["scene_key"])
        if (selected["selected_entry_relative_deg"] !=
                case["entry_relative_deg"] or not np.allclose(
                    selected["actor_start_xz_m"],
                    case["screened_actor_start_xz_m"], atol=1e-6)):
            raise ValueError("Entry differs from pre-outcome geometry screen")
    report = json.loads(study.REPORT.read_text())
    expected = [(c["id"], arm, False) for c in frozen["cases"]
                for arm in frozen["arms"]] + [(c["id"], "range360_early", True)
                for c in frozen["cases"]]
    rows = report["rows"]
    if len(rows) != len(expected) or sorted((r["case"], r["arm"], r["parked"])
            for r in rows) != sorted(expected):
        raise ValueError("Incomplete paired arm grid")
    all_rows = []
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    by_case = {c["id"]: [] for c in frozen["cases"]}
    for row in rows:
        by_case[row["case"]].append(row)
    for raw in frozen["cases"]:
        case = expanded_case(raw)
        source = json.loads((ROOT / case["source_capture"]).read_text())
        scene = TexturedCaptureScene(ROOT / case["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.),
            source_z_up=source.get("source_z_up", False))
        radii = {name: TexturedFloorObject(props / f"{name}.obj",
            scale=.012).footprint_radius_m for name in
            (case["target"], case["distractor"], case["hazard_object"])}
        for row in by_case[case["id"]]:
            all_rows.append(prior_audit.audit_row(case, row, scene,
                radii[case["hazard_object"]], radii))
    arms = {}
    for arm in frozen["arms"]:
        items = [x for x in all_rows if x["arm"] == arm and not x["parked"]]
        arms[arm] = {"runs": len(items),
            "alarms": sum(x["alarm_s"] is not None for x in items),
            "hazard_gaps_below_5cm": sum(x["min_hazard_gap_m"] < .05
                                         for x in items),
            "static_or_task_prop_contact": sum(x["min_static_gap_m"] < 0 or
                x["min_prop_gap_m"] < 0 for x in items),
            "floor_supported": sum(x["floor_supported"] for x in items),
            "stale_candidates_invalidated": sum(
                x["image_candidate_before_monitor"] and
                not x["image_valid_after_monitor"] for x in items)}
    totals = {"screened_rooms": 20, "supported_rooms": 19,
        "runs": len(all_rows),
        "parked_false_alarms": sum(x["alarm_s"] is not None for x in all_rows
                                   if x["parked"]),
        "max_monitor_packet_interval_s": max(x["monitor_max_packet_interval_s"]
                                               for x in all_rows),
        "by_arm": arms}
    AUDIT.write_text(json.dumps({"schema": "ape.item-inference-response-19-audit.v1",
        "precommit": relative(study.PRECOMMIT), "report": relative(study.REPORT),
        "totals": totals, "rows": all_rows,
        "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
