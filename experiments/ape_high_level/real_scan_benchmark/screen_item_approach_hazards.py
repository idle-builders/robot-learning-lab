"""Pre-screen moving-hazard entry and response lanes for 20 real rooms.

This is evaluator geometry only. It never supplies a lane, object position or
route to the actor; it selects feasible challenge cases before policy runs.
"""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_post_confirmation_approach_v2 as approach


BASE = approach.base
COHORT = ART / "integrated_owner_20_conditional_precommit.json"
REPORT = ART / "item_approach_hazard_lane_screen.json"
ENTRY_OFFSETS = (180., -90., 90., 0.)
INBOUND_RADII = (1., .8, .6, .4, .2)
ESCAPE_RADII = (.2, .4, .6)
ACTOR_RADIUS_M = .17
HAZARD_RADIUS_M = .06
STATIC_MARGIN_M = .05


def clearance(scene, xy: np.ndarray) -> dict:
    eye = np.array([xy[0], scene.floor_y + .42, xy[1]])
    angles = np.arange(-180., 180., 10., dtype=np.float32)
    rays = scene.range_fan(eye, 0., owner_xz=None, angles_deg=angles)
    finite = rays[np.isfinite(rays)]
    return {"floor_supported": bool(scene.floor_supported(*xy)),
        "nearest_two_plane_m": float(finite.min()) if len(finite) else None}


def screen_case(case: dict, index: int) -> dict:
    key = case["scene_key"]
    source = (ROOT /
        "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" /
        key)
    owner = json.loads((source / "owner_approach/trace.json").read_text())
    last = owner["trace"][-1]
    screen_data = json.loads((ROOT / case["screen"]).read_text())
    object_screen = next(x for x in screen_data["scenes"] if
                         x.get("scene_key") == key)
    scene = BASE.make_scene(case, object_screen)
    actual = np.asarray(last["evaluator_actual_xz_m"], dtype=float)
    heading = float(last["evaluator_actual_heading_deg"])
    candidates = []
    offsets = ENTRY_OFFSETS[index % 4:] + ENTRY_OFFSETS[:index % 4]
    for offset in offsets:
        angle = math.radians(heading + offset)
        direction = np.array([math.sin(angle), math.cos(angle)])
        inbound = [clearance(scene, actual + radius * direction)
                   for radius in INBOUND_RADII]
        escape = [clearance(scene, actual - radius * direction)
                  for radius in ESCAPE_RADII]
        required = ACTOR_RADIUS_M + HAZARD_RADIUS_M + STATIC_MARGIN_M
        def okay(points):
            return all(p["floor_supported"] and
                (p["nearest_two_plane_m"] is None or
                 p["nearest_two_plane_m"] >= required) for p in points)
        candidates.append({"entry_relative_deg": offset,
            "inbound": inbound, "escape": escape,
            "screen_pass": okay(inbound) and okay(escape)})
    selected = next((x for x in candidates if x["screen_pass"]), None)
    return {"scene_key": key, "capture_group": case["capture_group"],
        "actor_start_xz_m": actual.tolist(), "actor_heading_deg": heading,
        "selected_entry_relative_deg": selected["entry_relative_deg"]
            if selected else None,
        "candidates": candidates}


def main() -> None:
    cases = json.loads(COHORT.read_text())["cases"]
    if len(cases) != 20 or len({x["capture_group"] for x in cases}) != 20:
        raise ValueError("Expected 20 independent captures")
    rows = [screen_case(case, i) for i, case in enumerate(cases)]
    report = {"schema": "ape.item-approach-hazard-lane-screen.v1",
        "cohort": relative(COHORT), "cohort_sha256": sha(COHORT),
        "screen_order_relative_deg": ENTRY_OFFSETS,
        "rotated_first_choice_by_case_index": True,
        "inbound_radii_m": INBOUND_RADII, "escape_radii_m": ESCAPE_RADII,
        "required_nearest_two_plane_m": ACTOR_RADIUS_M + HAZARD_RADIUS_M +
            STATIC_MARGIN_M,
        "passed": sum(r["selected_entry_relative_deg"] is not None for r in rows),
        "rows": rows,
        "limits": "Evaluator static geometry and frozen prop placements only; no person motion, no policy outcome and no actor-visible route. The 0.06m hazard radius models a small prop, not a human body."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("passed", report["passed"], "/ 20")
    for row in rows:
        print(row["scene_key"], row["selected_entry_relative_deg"])


if __name__ == "__main__":
    main()
