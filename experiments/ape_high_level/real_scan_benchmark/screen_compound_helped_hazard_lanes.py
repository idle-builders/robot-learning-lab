"""Select second-entrant geometry before final-check hazard outcomes exist."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_compound_owner_point_recovery import case_rows, make_scene
from run_compound_owner_point_approach import REPORT as APPROACH_REPORT
from screen_item_approach_hazards import clearance


OUT = ART / "compound_helped_hazard_lane_screen.json"
OFFSETS = (180., -90., 90., 0.)
INBOUND = (1., .8, .6, .4, .2)
ESCAPE = (.2, .4, .6)
REQUIRED_M = .17 + .06 + .05


def main() -> None:
    cases = case_rows()
    prior = json.loads(APPROACH_REPORT.read_text())
    if len(cases) != 6 or len(prior["rows"]) != 6:
        raise ValueError("Expected six helped action endpoints")
    rows = []
    for index, case in enumerate(cases):
        found = [r for r in prior["rows"] if r["scene_key"] == case["scene_key"]]
        if len(found) != 1:
            raise ValueError("Missing action trace")
        trace = json.loads((ROOT / found[0]["trace"]).read_text())
        if trace["final_action"] != "owner_confirmed_visual_identity":
            raise ValueError("Final owner item not confirmed")
        endpoint = np.asarray(trace["walk"]["actual_final_xz_m"], float)
        yaw = float(trace["walk"]["actions"][-1]["evaluator_actual_yaw_deg"])
        scene = make_scene(case, True)
        ordered = OFFSETS[index % 4:] + OFFSETS[:index % 4]
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
                "inbound": inbound, "escape": escape, "supported": supported})
        selected = next((c for c in candidates if c["supported"]), None)
        row = {"case": case["scene_key"],
            "action_trace": found[0]["trace"],
            "endpoint_xz_m": endpoint.tolist(), "heading_deg": yaw,
            "selected_relative_bearing_deg":
                selected["relative_bearing_deg"] if selected else None,
            "candidates": candidates}
        rows.append(row)
        print(row["case"], row["selected_relative_bearing_deg"], flush=True)
    OUT.write_text(json.dumps({"schema":
        "ape.compound-helped-hazard-lane-screen.v1",
        "source_report": relative(APPROACH_REPORT),
        "source_report_sha256": sha(APPROACH_REPORT),
        "screen_order_relative_deg": OFFSETS,
        "inbound_radii_m": INBOUND, "escape_radii_m": ESCAPE,
        "required_nearest_two_plane_m": REQUIRED_M,
        "supported": sum(r["selected_relative_bearing_deg"] is not None
                         for r in rows), "rows": rows,
        "limits": "Evaluator-only static floor/room/task-prop geometry; no hazard, sensing or policy outcome informs selection."},
        indent=2) + "\n")
    print("supported", sum(r["selected_relative_bearing_deg"] is not None
                           for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
