"""Check swept wait corridors from a full actor-side two-height range scan."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_post_confirmation_approach_v2 as source
import screen_20_final_wait_lanes as screen


OUT = ART / "scanned_wait_corridor_20.json"
SCAN_ANGLES = np.arange(-180.,180.,5.,dtype=np.float32)


def certified(samples: list[np.ndarray], ranges: np.ndarray,
              origin: np.ndarray, yaw: float, required: float) -> bool:
    angles = np.radians(yaw + SCAN_ANGLES)
    vectors = np.stack([np.sin(angles),np.cos(angles)],axis=1)
    hit_points = np.concatenate([origin + np.where(
        np.isfinite(ranges[height,:,None]),ranges[height,:,None],0.) * vectors
        for height in range(2)],axis=0)
    valid = np.concatenate([np.isfinite(ranges[height]) for height in range(2)])
    hits = hit_points[valid]
    if not len(hits):
        return False
    for sample in samples:
        delta = sample-origin
        distance = float(np.linalg.norm(delta))
        bearing = (math.degrees(math.atan2(delta[0],delta[1])) - yaw + 180.)%360.-180.
        index = int(np.argmin(np.abs((SCAN_ANGLES-bearing+180.)%360.-180.)))
        rays = ranges[:,index]
        if not np.all(np.isfinite(rays)) or np.min(rays) < distance+required:
            return False
        if float(np.min(np.linalg.norm(hits-sample,axis=1))) < required:
            return False
    return True


def main() -> None:
    reference = json.loads(screen.OUT.read_text())
    cases = {c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    rows = []
    for ref in reference["rows"]:
        case = cases[ref["scene_key"]]
        placement_doc = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in placement_doc["scenes"] if x[
            "scene_key"] == ref["scene_key"])
        scene = source.base.make_scene(case,placement)
        origin = np.asarray(ref["endpoint_xz_m"],float)
        yaw = ref["heading_deg"]
        eye = np.array([origin[0],scene.floor_y+.42,origin[1]])
        rays = scene.range_fan(eye,yaw,owner_xz=None,angles_deg=SCAN_ANGLES)
        bearings = []
        for offset in screen.OFFSETS:
            angle = math.radians(yaw+offset)
            direction = np.array([math.sin(angle),math.cos(angle)])
            samples = [origin+d*direction for d in screen.INBOUND_M]
            samples += [origin-d*direction for d in screen.ESCAPE_M]
            accepted = certified(samples,rays,origin,yaw,
                                 reference["required_nearest_two_plane_m"])
            bearings.append({"relative_bearing_deg":offset,
                "actor_scanned_corridor":accepted,
                "evaluator_supported":offset in ref[
                    "supported_relative_bearings_deg"]})
        rows.append({"scene_key":ref["scene_key"],
            "capture_xz_m":origin.tolist(),"capture_heading_deg":yaw,
            "scan_angles_deg":SCAN_ANGLES.tolist(),
            "low_range_m":[float(x) if np.isfinite(x) else None for x in rays[0]],
            "mid_range_m":[float(x) if np.isfinite(x) else None for x in rays[1]],
            "bearings":bearings,
            "actor_has_corridor":any(b["actor_scanned_corridor"]
                                     for b in bearings),
            "evaluator_has_corridor":bool(ref[
                "supported_relative_bearings_deg"])})
        print(ref["scene_key"],rows[-1]["actor_has_corridor"],
              rows[-1]["evaluator_has_corridor"],flush=True)
    flat = [b for row in rows for b in row["bearings"]]
    result = {"schema":"ape.scanned-wait-corridor-20.v1",
        "source_screen":relative(screen.OUT),
        "source_screen_sha256":sha(screen.OUT),
        "scan_increment_deg":5.,
        "actor_certified_rooms":sum(r["actor_has_corridor"] for r in rows),
        "false_positive_bearings":sum(b["actor_scanned_corridor"] and not
            b["evaluator_supported"] for b in flat),
        "missed_supported_bearings":sum(b["evaluator_supported"] and not
            b["actor_scanned_corridor"] for b in flat),
        "rows":rows,"limits":"Ideal static two-height 360-degree range at the held pose; evaluator still scores floor and room/prop geometry. No measured mount/noise, moving person, source outage, swept action or fresh visual identity tested."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("actor rooms",result["actor_certified_rooms"],
          "false positive bearings",result["false_positive_bearings"])


if __name__ == "__main__":
    main()
