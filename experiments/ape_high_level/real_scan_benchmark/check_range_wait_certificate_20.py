"""Compare an actor-observed all-around wait gate with evaluator corridors."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_post_confirmation_approach_v2 as source
import screen_20_final_wait_lanes as screen


OUT = ART / "range_wait_certificate_20.json"


def main() -> None:
    expected = json.loads(screen.OUT.read_text())
    cases = {c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    rows = []
    for row in expected["rows"]:
        case = cases[row["scene_key"]]
        placement_file = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in placement_file["scenes"] if x[
            "scene_key"] == row["scene_key"])
        scene = source.base.make_scene(case,placement)
        xy = np.asarray(row["endpoint_xz_m"],float)
        eye = np.array([xy[0],scene.floor_y+.42,xy[1]])
        yaw = row["heading_deg"]
        angles = np.asarray(screen.OFFSETS,np.float32)
        rays = scene.range_fan(eye,yaw,owner_xz=None,angles_deg=angles)
        required = expected["required_nearest_two_plane_m"]
        gate = []
        for index,bearing in enumerate(angles):
            reverse_index = next(j for j,other in enumerate(angles) if
                abs((float(other)-float(bearing)+180.)%360.-180.)==180.)
            inbound = rays[:,index]
            escape = rays[:,reverse_index]
            accepted = bool(np.all(np.isfinite(inbound)) and
                np.all(np.isfinite(escape)) and
                np.min(inbound) >= 1.+required and
                np.min(escape) >= .6+required)
            gate.append({"entry_relative_bearing_deg":float(bearing),
                "inbound_low_mid_m":inbound.tolist(),
                "escape_low_mid_m":escape.tolist(),
                "actor_range_certificate":accepted,
                "evaluator_lane_supported":float(bearing) in row[
                    "supported_relative_bearings_deg"]})
        rows.append({"scene_key":row["scene_key"],
            "actor_has_certificate":any(g["actor_range_certificate"] for g in gate),
            "evaluator_has_lane":bool(row["supported_relative_bearings_deg"]),
            "bearings":gate})
        print(row["scene_key"],rows[-1]["actor_has_certificate"],
              rows[-1]["evaluator_has_lane"],flush=True)
    flat = [g for r in rows for g in r["bearings"]]
    result = {"schema":"ape.range-wait-certificate-20.v1",
        "source_screen":relative(screen.OUT),"source_screen_sha256":sha(screen.OUT),
        "inbound_range_requirement_m":1.+expected["required_nearest_two_plane_m"],
        "escape_range_requirement_m":.6+expected["required_nearest_two_plane_m"],
        "actor_certified_rooms":sum(r["actor_has_certificate"] for r in rows),
        "actor_certified_false_positive_bearings":sum(g[
            "actor_range_certificate"] and not g["evaluator_lane_supported"]
            for g in flat),
        "evaluator_supported_bearings_missed":sum(g[
            "evaluator_lane_supported"] and not g["actor_range_certificate"]
            for g in flat),
        "rows":rows,"limits":"Ideal instantaneous two-height all-around range at held pose; no floor or swept-body certification, no source age or dynamic person, no alternate-pose action. Evaluator lane is not a safety ground truth for a human."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("actor certificates",result["actor_certified_rooms"],
          "false positive bearings",result["actor_certified_false_positive_bearings"])


if __name__ == "__main__":
    main()
