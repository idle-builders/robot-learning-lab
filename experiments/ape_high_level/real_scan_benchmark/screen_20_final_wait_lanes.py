"""Measure how often the unchanged final item pose has an entry/escape lane."""

from __future__ import annotations

import json
import math

import numpy as np

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from screen_item_approach_hazards import clearance
import run_post_confirmation_approach_v2 as source


OUT = ART / "final_wait_lane_screen_20.json"
OFFSETS = (180., -90., 90., 0.)
INBOUND_M = (1., .8, .6, .4, .2)
ESCAPE_M = (.2, .4, .6)
BODY_RADIUS_M = .17
GEOMETRY_MARGIN_M = .05


def main() -> None:
    report = json.loads(source.REPORT.read_text())
    cases = json.loads((ART / "integrated_owner_20_conditional_precommit.json").read_text())["cases"]
    if len(cases) != 20 or len(report["rows"]) != 20:
        raise ValueError("Expected 20 finished item-view actions")
    by_key = {r["scene_key"]: r for r in report["rows"]}
    hazard_radius = TexturedFloorObject(ROOT /
        ".cache/ape-high-level/hope_meshes_eval/Popcorn.obj",scale=.012).footprint_radius_m
    required = BODY_RADIUS_M + hazard_radius + GEOMETRY_MARGIN_M
    rows = []
    for case in cases:
        key = case["scene_key"]
        trace = json.loads((ROOT / by_key[key]["trace"]).read_text())
        if trace["action_result"] != "observed_target_standoff" or trace[
                "reconfirmation_action"] != "owner_confirmed_visual_identity":
            raise ValueError("Endpoint was not a completed item view")
        endpoint = np.asarray(trace["actions"][-1]["evaluator_actual_xz_m"],float)
        yaw = float(trace["actions"][-1]["evaluator_actual_heading_deg"])
        screen = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in screen["scenes"] if x["scene_key"] == key)
        scene = source.base.make_scene(case, placement)
        candidates = []
        for offset in OFFSETS:
            angle = math.radians(yaw + offset)
            direction = np.array([math.sin(angle),math.cos(angle)])
            inbound = [clearance(scene,endpoint+d*direction) for d in INBOUND_M]
            escape = [clearance(scene,endpoint-d*direction) for d in ESCAPE_M]
            supported = all(p["floor_supported"] and
                (p["nearest_two_plane_m"] is None or
                 p["nearest_two_plane_m"] >= required)
                for p in (*inbound,*escape))
            candidates.append({"relative_bearing_deg":offset,
                "supported":supported,"inbound":inbound,"escape":escape})
        supported = [c["relative_bearing_deg"] for c in candidates if c["supported"]]
        rows.append({"scene_key":key,"capture_group":case["capture_group"],
            "action_trace":by_key[key]["trace"],"endpoint_xz_m":endpoint.tolist(),
            "heading_deg":yaw,"supported_relative_bearings_deg":supported,
            "candidates":candidates})
        print(key, supported, flush=True)
    OUT.write_text(json.dumps({"schema":"ape.final-wait-lane-screen-20.v1",
        "source_report":relative(source.REPORT),
        "source_report_sha256":sha(source.REPORT),
        "hazard_prop":"Popcorn","rendered_hazard_radius_m":hazard_radius,
        "body_radius_m":BODY_RADIUS_M,"geometry_margin_m":GEOMETRY_MARGIN_M,
        "required_nearest_two_plane_m":required,"inbound_radii_m":INBOUND_M,
        "escape_radii_m":ESCAPE_M,"screen_bearings_deg":OFFSETS,
        "supported":sum(bool(r["supported_relative_bearings_deg"]) for r in rows),
        "rows":rows,"limits":"Evaluator-only static scene and task-prop geometry at previously reached poses. This is a policy design diagnostic; it does not certify a planned waiting pose, moving-person safety, or physical ape stop."},indent=2)+"\n")
    print("supported",sum(bool(r["supported_relative_bearings_deg"]) for r in rows),"/",len(rows))


if __name__ == "__main__":
    main()
