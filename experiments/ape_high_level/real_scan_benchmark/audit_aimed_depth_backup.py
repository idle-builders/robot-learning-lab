"""Recompute aimed-depth change evidence and ideal response from saved frames."""

from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative
import run_cross_room_inference_response as motion
import probe_aimed_depth_backup as trial
import audit_shared_world_return as video_audit


OUT = ART / "aimed_depth_backup_audit.json"


def audit_one(row: dict, case: dict, lane: dict) -> dict:
    bearing = row["last_range_track_bearing_deg"]
    if bearing is not None and bearing not in (float(x) for x in
                                            np.arange(-180., 181., 5.)):
        raise ValueError("Head aim did not come from a low-range ray")
    if row["head_yaw_supported"] != (bearing is not None and
                                     abs(bearing) <= trial.MAX_YAW_DEG):
        raise ValueError("Head yaw support contradicts frozen limit")
    if not row["head_yaw_supported"]:
        if row["samples"] or row["video"] is not None:
            raise ValueError("Unavailable rear view has rendered evidence")
        return {"case": row["case"], "head_yaw_supported": False,
                "first_alarm_s": None, "min_gap_m": None}
    baseline = np.asarray(Image.open(ROOT / row["baseline_depth"]),
                          np.float32) / 1000.
    baseline[baseline == 0] = np.nan
    if video_audit.decoded_frames(ROOT / row["video"]) != len(trial.SAMPLE_TIMES):
        raise ValueError("Aimed-depth video/trace frame mismatch")
    first_alarm = None
    for sample, t in zip(row["samples"], trial.SAMPLE_TIMES):
        if sample["capture_s"] != t:
            raise ValueError("Depth capture time changed")
        depth = np.asarray(Image.open(ROOT / sample["depth"]),
                           np.float32) / 1000.
        depth[depth == 0] = np.nan
        if not (ROOT / sample["rgb"]).exists():
            raise ValueError("Missing RGB frame")
        valid = np.isfinite(depth) & np.isfinite(baseline)
        changed = valid & (baseline - depth > .08) & (depth < .9)
        count = int(changed.sum())
        # Quantized millimetres can move a threshold-edge pixel. Alarms must
        # still reproduce exactly after that quantization.
        if abs(count - sample["changed_pixels"]) > 3:
            raise ValueError("Changed-depth pixel count does not replay")
        alarm = count >= trial.CHANGED_PIXELS
        if alarm != sample["alarm"]:
            raise ValueError("Aimed-depth alarm does not replay")
        if first_alarm is None and alarm:
            first_alarm = t
    if first_alarm != row["first_depth_alarm_s"]:
        raise ValueError("First depth alarm differs from saved trace")
    start = np.asarray(lane["response_endpoint_xz_m"], float)
    yaw = float(lane["actor_heading_deg"])
    entry = math.radians(yaw + lane["selected_relative_bearing_deg"])
    incoming = np.array([math.sin(entry), math.cos(entry)])
    look = math.radians(yaw + bearing)
    opposite = np.array([-math.sin(look), -math.cos(look)])
    prop = TexturedFloorObject(ROOT / ".cache/ape-high-level/hope_meshes_eval" /
                               f"{case['hazard_object']}.obj", scale=.012)
    gaps = []
    for t in np.arange(0., 4.51, .02):
        actor = motion.actor_xy(start, opposite if first_alarm is not None else
                                None, first_alarm, float(t))
        hazard = motion.incoming_xy(start, incoming, float(t))
        gaps.append(motion.proxy_gap(actor, hazard, prop.footprint_radius_m))
    if not np.isclose(min(gaps), row["min_proxy_gap_with_depth_response_m"],
                      atol=1e-6):
        raise ValueError("Aimed-depth proxy response does not replay")
    return {"case": row["case"], "head_yaw_supported": True,
        "first_alarm_s": first_alarm, "min_gap_m": min(gaps),
        "gap_under_5cm": min(gaps) < .05}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    lanes = {x["case"]: x for x in json.loads(trial.LANES.read_text())["rows"]}
    cases = {x["id"]: x for x in trial.cohort.case_rows()}
    rows = [audit_one(row, cases[row["case"]], lanes[row["case"]])
            for row in report["rows"]]
    result = {"schema": "ape.aimed-depth-backup-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "aimable": sum(r["head_yaw_supported"] for r in rows),
        "aimed_alarms": sum(r["first_alarm_s"] is not None for r in rows),
        "aimed_gap_breaches": sum(r.get("gap_under_5cm", False) for r in rows),
        "unviewable_rear": sum(not r["head_yaw_supported"] for r in rows),
        "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
