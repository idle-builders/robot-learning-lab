"""Pre-outcome second-entrant geometry from saved response endpoints."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_item_inference_response_19 as cohort
import run_post_confirmation_approach_v2 as approach
from screen_item_approach_hazards import clearance
import run_shared_world_return as previous


OUT = ART / "monitored_return_lane_screen.json"
OFFSETS = (180., -90., 90., 0.)
INBOUND = (1., .8, .6, .4, .2)
ESCAPE = (.2, .4, .6)
REQUIRED_M = .17 + .06 + .05


def main() -> None:
    earlier = json.loads(previous.REPORT.read_text())
    by_case = {row["case"]: row for row in earlier["rows"]}
    cases = cohort.case_rows()
    if len(cases) != 19 or len(by_case) != 19:
        raise ValueError("Expected complete first-response cohort")
    rows = []
    for i, case in enumerate(cases):
        trace = json.loads((ROOT / by_case[case["id"]]["trace"]).read_text())
        endpoint = np.asarray(trace["response_endpoint_xz_m"], float)
        screen = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in screen["scenes"] if
                         x["scene_key"] == case["scene_key"])
        scene = approach.base.make_scene(case, placement)
        summary = json.loads((ROOT / case["teaching_summary"]).read_text())
        stop = json.loads((ROOT / summary["teaching"]["owner_approach_trace"]).read_text())["trace"][-1]
        yaw = float(stop["evaluator_actual_heading_deg"])
        ordered = OFFSETS[i % 4:] + OFFSETS[:i % 4]
        candidates = []
        for offset in ordered:
            bearing = math.radians(yaw + offset)
            direction = np.array([math.sin(bearing), math.cos(bearing)])
            inbound = [clearance(scene, endpoint + radius * direction)
                       for radius in INBOUND]
            escape = [clearance(scene, endpoint - radius * direction)
                      for radius in ESCAPE]
            supported = all(p["floor_supported"] and
                (p["nearest_two_plane_m"] is None or
                 p["nearest_two_plane_m"] >= REQUIRED_M)
                for p in (*inbound, *escape))
            candidates.append({"relative_bearing_deg": offset,
                "inbound": inbound, "escape": escape,
                "supported": supported})
        selected = next((c for c in candidates if c["supported"]), None)
        rows.append({"case": case["id"],
            "response_endpoint_xz_m": endpoint.tolist(),
            "actor_heading_deg": yaw,
            "selected_relative_bearing_deg": (selected["relative_bearing_deg"]
                                               if selected else None),
            "candidates": candidates})
        print(case["id"], rows[-1]["selected_relative_bearing_deg"], flush=True)
    result = {"schema": "ape.monitored-return-lane-screen.v1",
        "source_report": relative(previous.REPORT),
        "source_report_sha256": sha(previous.REPORT),
        "screen_order_relative_deg": OFFSETS,
        "inbound_radii_m": INBOUND, "escape_radii_m": ESCAPE,
        "required_nearest_two_plane_m": REQUIRED_M,
        "passed": sum(x["selected_relative_bearing_deg"] is not None
                      for x in rows),
        "rows": rows,
        "limits": "Evaluator-only static scene and task-prop geometry; no action, detector outcome or policy score informed lane selection."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("supported", result["passed"], "/", len(rows))


if __name__ == "__main__":
    main()
