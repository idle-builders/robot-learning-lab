"""Audit the paired 20-room clearance-sensor action trials."""

from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from rgbd_range_projection import projected_depth_fan, dual_view_fan
import run_item_approach_sensor_20 as study


AUDIT = ART / "item_approach_sensor_20_audit.json"


def decoded_frames(path):
    video = cv2.VideoCapture(str(path))
    frames = 0
    while video.read()[0]:
        frames += 1
    video.release()
    return frames


def saved_depth(path: str) -> np.ndarray:
    millimetres = np.asarray(Image.open(ROOT / path), dtype=np.float32)
    return np.where(millimetres > 0., millimetres / 1000., np.nan)


def reprojected_packet(action: dict, arm: str) -> np.ndarray | None:
    if arm == "separate_range":
        return None
    upright = saved_depth(action["depth"])
    if arm == "upright_rgbd":
        return projected_depth_fan(upright, study.ANGLES)
    down = saved_depth(action["down_depth"])
    packet = dual_view_fan(upright, down, study.ANGLES)
    for sweep in action.get("head_sweep", []):
        other = dual_view_fan(saved_depth(sweep["depth"]),
            saved_depth(sweep["down_depth"]),
            study.ANGLES - sweep["pan_deg"])
        packet = np.fmin(packet, other)
    return packet


def audit_row(row: dict) -> dict:
    trace = json.loads((ROOT / row["trace"]).read_text())
    key, arm = row["scene_key"], row["arm"]
    if trace["scene_key"] != key or trace["arm"] != arm:
        raise ValueError("Trace identity mismatch")
    actions = trace["actions"]
    if not actions or len(actions) > study.base.MAX_ACTIONS + 1:
        raise ValueError("Action count invalid")
    if decoded_frames(ROOT / trace["video"]) != len(actions):
        raise ValueError("RGB video does not match actions")
    steps = 0
    for i, action in enumerate(actions):
        if action["frame"] != i or abs(action["capture_time_s"] - .2 * i) > 1e-6:
            raise ValueError("Action timestamps misaligned")
        if not (ROOT / action["depth"]).is_file():
            raise ValueError("Upright depth frame missing")
        if arm in ("dual_rgbd", "dual_rgbd_active_pan") and not (
                ROOT / action["down_depth"]).is_file():
            raise ValueError("Downward depth frame missing")
        if arm not in ("dual_rgbd", "dual_rgbd_active_pan") and \
                action["down_depth"] is not None:
            raise ValueError("Unexpected downward depth")
        if arm == "dual_rgbd_active_pan":
            for sweep in action["head_sweep"]:
                if sweep["pan_deg"] not in (-20., 20.):
                    raise ValueError("Unexpected head pan")
                for field in ("rgb", "depth", "down_rgb", "down_depth"):
                    if not (ROOT / sweep[field]).is_file():
                        raise ValueError("Head-pan RGB-D frame missing")
        elif action.get("head_sweep"):
            raise ValueError("Head sweep in non-active arm")
        packet = action["sensor_packet_m"]
        if len(packet) != 2 or any(len(p) != len(study.ANGLES) for p in packet):
            raise ValueError("Sensor packet shape changed")
        front = [v for plane in packet for v, use in zip(plane, study.FRONT) if use]
        unknown = sum(v is None for v in front)
        if unknown != action["sensor_health"]["unknown_front_sectors"]:
            raise ValueError("Unknown coverage count wrong")
        decision = action["decision"]
        amount = action["amount"]
        if arm != "separate_range" and decision in ("forward", "hold"):
            replay = reprojected_packet(action, arm)[:, study.FRONT]
            saved = np.asarray([[np.nan if value is None else np.inf
                if value == "clear" else float(value)
                for value, use in zip(plane, study.FRONT) if use]
                for plane in packet])
            if not np.array_equal(np.isnan(replay), np.isnan(saved)):
                raise ValueError("Saved depth does not reproduce unknown sectors")
            if not np.array_equal(np.isinf(replay), np.isinf(saved)):
                raise ValueError("Saved depth does not reproduce clear sectors")
            finite = np.isfinite(replay) & np.isfinite(saved)
            if finite.any() and np.max(np.abs(replay[finite] - saved[finite])) > .03:
                raise ValueError("Saved depth does not reproduce obstacle range")
        if decision == "forward":
            if unknown or not (0. < amount <= study.base.STEP_M + 1e-6):
                raise ValueError("Forward move lacked full clearance evidence")
            values = [float("inf") if v == "clear" else float(v) for v in front]
            if min(values) < study.base.BODY_RADIUS_M + \
                    study.base.SENSOR_MARGIN_M + amount - 1e-6:
                raise ValueError("Forward move violated observed clearance")
            if not action.get("evaluator_motion_accepted"):
                raise ValueError("Simulator rejected forward move")
            steps += 1
        elif decision == "turn":
            if abs(amount) > study.base.TURN_DEG + 1e-6:
                raise ValueError("Turn limit exceeded")
        elif decision == "hold":
            if i != len(actions) - 1:
                raise ValueError("Motion after hold")
            if action["reason"] == "forward_clearance_unknown" and not unknown:
                raise ValueError("False unknown-clearance stop")
        else:
            raise ValueError("Unknown action")
    if row["accepted_forward_steps"] != steps:
        raise ValueError("Report step count differs")
    if row["exact_baseline_endpoint"]:
        if row["result"] != "observed_target_standoff" or not row["owner_reconfirmed"]:
            raise ValueError("Baseline endpoint reused incorrectly")
        baseline = json.loads((study.v2.ASSETS / key / "trace.json").read_text())
        if baseline["reconfirmation_action"] != "owner_confirmed_visual_identity":
            raise ValueError("Source owner answer not confirmed")
        last = actions[-1]
        old_last = baseline["actions"][-1]
        if not np.allclose(last["evaluator_actual_xz_m"],
                           old_last["evaluator_actual_xz_m"], atol=1e-6):
            raise ValueError("Reused endpoint differs")
        if not np.allclose(last["actor_xz_m"],
                           old_last["actor_pose_world_xz_m"], atol=1e-6):
            raise ValueError("Reused actor pose differs")
    elif row["owner_reconfirmed"]:
        raise ValueError("Owner answer reused without exact endpoint")
    return {"scene_key": key, "capture_group": row["capture_group"],
        "arm": arm, "result": row["result"],
        "accepted_forward_steps": steps,
        "unknown_stop": row["result"] == "forward_clearance_unknown",
        "exact_endpoint_and_owner_confirmed": row["exact_baseline_endpoint"] and
            row["owner_reconfirmed"],
        "final_actual_target_gap_m": row["final_actual_target_gap_m"],
        "frames": len(actions), "video": row["video"], "trace": row["trace"]}


def main() -> None:
    precommit = json.loads(study.PRECOMMIT.read_text())
    if sha(ROOT / precommit["cohort"]) != precommit["cohort_sha256"] or \
            sha(ROOT / precommit["baseline_audit"]) != precommit["baseline_audit_sha256"]:
        raise ValueError("Frozen source changed")
    cases = json.loads(study.COHORT.read_text())["cases"]
    if len(cases) != 20 or len({c["capture_group"] for c in cases}) != 20:
        raise ValueError("Capture cohort invalid")
    for screen in precommit["screens"]:
        if sha(ROOT / screen["screen"]) != screen["sha256"]:
            raise ValueError("Frozen screen changed")
    report = json.loads(study.REPORT.read_text())
    rows = report["rows"]
    if len(rows) != len(cases) * len(study.ARMS) or [
            (x["scene_key"], x["arm"]) for x in rows] != [
            (c["scene_key"], a) for c in cases for a in study.ARMS]:
        raise ValueError("Paired arm grid incomplete")
    audited = [audit_row(row) for row in rows]
    totals = {arm: {"rooms": 20,
        "observed_standoff": sum(x["result"] == "observed_target_standoff"
            for x in audited if x["arm"] == arm),
        "owner_reconfirmed_at_exact_endpoint": sum(
            x["exact_endpoint_and_owner_confirmed"] for x in audited
            if x["arm"] == arm),
        "unknown_stops": sum(x["unknown_stop"] for x in audited
            if x["arm"] == arm),
        "accepted_forward_steps": sum(x["accepted_forward_steps"] for x in
            audited if x["arm"] == arm)} for arm in study.ARMS}
    AUDIT.write_text(json.dumps({"schema": "ape.item-approach-sensor-20-audit.v1",
        "precommit": relative(study.PRECOMMIT), "report": relative(study.REPORT),
        "totals": totals, "rows": audited}, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
