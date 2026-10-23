"""Audit live aimed-depth backup timing, saved frames and owner outcomes."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative
from run_conditional_search_rear_hazard import ANGLES
import audit_monitored_return_stress as base_audit
import audit_shared_world_return as video_audit
import run_monitored_return_depth_backup as trial


OUT = ART / "monitored_return_depth_backup_audit.json"


def audit_one(row: dict, case: dict, lane: dict,
              expected_arm: str = trial.ARM) -> dict:
    scored = base_audit.audit_one(case, row)
    episode = json.loads((ROOT / row["trace"]).read_text())
    phase = json.loads((ROOT / episode["second_entry_trace"]).read_text())
    if phase["arm"] != expected_arm or not phase["image_invalidated"]:
        raise ValueError("Backup phase did not invalidate the interrupted image")
    if phase["source_health_lost_s"] is None:
        raise ValueError("Outage was not observed")
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if
                     x["scene_key"] == case["scene_key"])
    scene = base_audit.approach.base.make_scene(case, placement)
    start = np.asarray(phase["origin_actor_xz_m"], float)
    eye = np.array([start[0], scene.floor_y + .42, start[1]])
    yaw = float(lane["actor_heading_deg"])
    baseline = scene.range_fan(eye, yaw, owner_xz=None,
                               angles_deg=ANGLES)[0]
    first = next(p for p in phase["monitor_packets"] if p["low_range_m"] is not None)
    rays = np.asarray([np.nan if x is None else x for x in
                       first["low_range_m"]], float)
    with np.errstate(invalid="ignore"):
        track = (baseline - rays > .02) & (rays < 1.3)
    expected_bearing = None
    if np.any(track):
        ids = np.flatnonzero(track)
        expected_bearing = float(ANGLES[ids[np.argmin(rays[ids])]])
    if phase["backup_head_track_bearing_deg"] != expected_bearing:
        raise ValueError("Head aim was not derived from the healthy range packet")
    aimable = expected_bearing is not None and abs(expected_bearing) <= 90.
    if phase["backup_head_yaw_supported"] != aimable:
        raise ValueError("Head yaw support contradicts fixed limit")
    samples = phase["backup_head_samples"]
    if not aimable and (samples or phase["backup_head_video"] is not None):
        raise ValueError("Rear lane was given a synthetic camera view")
    if samples:
        if video_audit.decoded_frames(ROOT / phase["backup_head_video"]) != len(samples):
            raise ValueError("Head-depth video and trace diverge")
    previous = None
    first_depth_alarm = None
    for sample in samples:
        depth = np.asarray(Image.open(ROOT / sample["depth"]),
                           np.float32) / 1000.
        depth[depth == 0] = np.nan
        if not (ROOT / sample["rgb"]).exists():
            raise ValueError("Missing aimed RGB frame")
        count = None
        if previous is not None:
            valid = np.isfinite(previous) & np.isfinite(depth)
            count = int((valid & (previous - depth > .08) &
                         (depth < .9)).sum())
        if (count is None) != (sample["changed_pixels"] is None):
            raise ValueError("Head-depth frame history differs")
        if count is not None and abs(count - sample["changed_pixels"]) > 3:
            raise ValueError("Head-depth change count does not replay")
        alarm = count is not None and count >= trial.base.BACKUP_CHANGED_PIXELS
        if alarm != sample["alarm"]:
            raise ValueError("Head-depth alarm does not replay")
        if first_depth_alarm is None and alarm:
            first_depth_alarm = sample
        previous = depth
    if first_depth_alarm is not None:
        if phase["alarm_source"] != "aimed_head_depth":
            raise ValueError("Depth alarm did not issue the response")
        if not np.isclose(phase["alarm_delivery_s"],
                          first_depth_alarm["delivery_wall_s"], atol=1e-6):
            raise ValueError("Response used inference time rather than delivered depth")
        if phase["alarm_relative_angle_deg"] != expected_bearing:
            raise ValueError("Response did not use the prior observed bearing")
    return {**scored,
        "head_aimable": aimable,
        "head_samples": len(samples),
        "head_alarm_s": (first_depth_alarm["delivery_wall_s"]
                         if first_depth_alarm else None),
        "head_alarm_source": phase["alarm_source"] == "aimed_head_depth"}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    lanes = {x["case"]: x for x in json.loads(trial.base.LANES.read_text())["rows"]}
    cases = {x["id"]: x for x in trial.cohort.case_rows()}
    rows = [audit_one(row, cases[row["case"]], lanes[row["case"]])
            for row in report["rows"]]
    result = {"schema": "ape.monitored-return-depth-backup-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "head_aimable": sum(r["head_aimable"] for r in rows),
        "head_alarms": sum(r["head_alarm_source"] for r in rows),
        "modeled_gap_breaches": sum(r["gap_under_5cm"] for r in rows),
        "aimable_gap_breaches": sum(r["head_aimable"] and
                                    r["gap_under_5cm"] for r in rows),
        "static_or_prop_contacts": sum(r["min_static_gap_m"] < 0 or
            r["min_task_prop_gap_m"] < 0 or not
            r["floor_supported_throughout"] for r in rows),
        "owner_confirmed": sum(r["outcome"] == "fresh_owner_confirmed"
                               for r in rows),
        "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
