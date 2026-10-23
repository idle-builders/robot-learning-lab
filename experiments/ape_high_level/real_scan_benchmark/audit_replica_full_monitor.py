"""Independent paired audit of whole-task range health in Replica rooms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "assets/real_indoor_scenes/replica_owner_monitored_20261015"
REPORT = ROOT / "artifacts/ape-replica-interactive-20261015/owner_full_monitor_audit.json"
ROOMS = ("hotel_0", "apartment_0")
ARMS = ("none", "walk", "final")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape[:2] == (480, 640), path
        count += 1
    capture.release()
    return count


def check_visual_stage(folder: Path) -> dict:
    trace = json.loads((folder / "trace.json").read_text())
    frames = video_frames(folder / "first_person.mp4")
    assert frames == trace["frames"]
    for index in range(frames):
        rgb = np.asarray(Image.open(folder / f"rgb_{index:03d}.png"))
        depth = np.asarray(Image.open(folder / f"depth_{index:03d}.png"))
        assert rgb.shape == (480, 640, 3)
        assert depth.shape == (480, 640) and depth.dtype == np.uint16
        assert np.count_nonzero(depth) > depth.size // 2
    return {"action": trace["action"], "frames": frames,
        "wrong_confident": trace["wrong_confident_candidate"],
        "selected_target_iou": (trace.get("verification") or {}).get(
            "evaluator_selected_target_iou")}


def check_arm(room: str, arm: str) -> dict:
    folder = RUNS / room / arm
    summary = json.loads((folder / "summary.json").read_text())
    scenario = json.loads((folder / "scenario.json").read_text())
    source = json.loads((folder / "independent_range/packets.json").read_text())
    walk = json.loads((folder / "walk/trace.json").read_text())
    packets = source["packets"]
    assert summary["scene"] == room and summary["range_gap_stage"] == arm
    assert scenario["range_gap_stage"] == source["fault"] == arm
    assert all(x["local_floor_y_m"] is not None for x in
               scenario["support_screen"].values())
    assert [p["index"] for p in packets] == list(range(len(packets)))
    assert all(b["capture_wall_s"] > a["capture_wall_s"] for a, b in
               zip(packets, packets[1:]))
    assert all(p["delivery_wall_s"] >= p["capture_wall_s"] for p in packets)
    assert len(list((folder / "rgb_requests").glob("frame_*.jpg"))) == summary[
        "rgb_requests"]
    stages = {name: check_visual_stage(folder / name) for name in
              ("find_present", "find_absent")}
    assert stages["find_present"]["action"] == "candidate_to_owner_for_confirmation"
    assert stages["find_absent"]["action"] == "report_unidentified"
    assert not any(stage["wrong_confident"] for stage in stages.values())
    assert summary["find_source_health"]["authorized"]
    assert summary["find_source_health"]["max_capture_interval_s"] <= .2
    assert video_frames(folder / "walk/first_person.mp4") == len(walk["actions"])
    assert len(walk["actions"]) <= walk["action_budget"] + 1
    accepted = [a for a in walk["actions"] if a["decision"] == "forward" and
                a.get("evaluator_motion_accepted")]
    assert len(accepted) == summary["walk_accepted_forward_steps"]
    assert all(a["source_guard"]["authorized"] and
               a["source_guard"]["last_packet_age_s"] <= .2 and
               0 < a["amount"] <= .04 for a in accepted)
    missing = [p for p in packets if p["source_health"] == "missing"]
    if arm == "walk":
        assert summary["walk_result"] == "source_gap"
        assert len(accepted) == 3
        assert summary["final_action"] is None
        assert summary["walk_source_health"]["source_gap_seen"]
        assert walk["actions"][-1]["decision"] == "hold"
        assert walk["actions"][-1]["source_guard"]["reason"] == "source_gap"
        assert missing
        response_s = (walk["actions"][-1]["source_guard"]["checked_wall_s"] -
                      missing[0]["capture_wall_s"])
        assert 0 <= response_s <= .2
        final_stage = None
    else:
        assert summary["walk_result"] == "observed_target_standoff"
        assert summary["walk_source_health"]["authorized"]
        final_stage = check_visual_stage(folder / "final_view")
        assert final_stage["action"] == "candidate_to_owner_for_confirmation"
        assert final_stage["selected_target_iou"] >= .25
        response_s = None
        if arm == "none":
            assert summary["final_source_health"]["authorized"]
            assert summary["final_action"] == final_stage["action"]
            assert not missing
        else:
            assert summary["final_action_raw"] == final_stage["action"]
            assert summary["final_action"] == "withheld_range_source_unhealthy"
            assert summary["final_source_health"]["source_gap_seen"]
            assert missing
    return {"arm": arm, "scenario": scenario, "stages": stages,
        "walk_result": summary["walk_result"],
        "accepted_forward_steps": len(accepted),
        "effective_final_action": summary["final_action"],
        "final_visual_action": final_stage["action"] if final_stage else None,
        "independent_packets": len(packets), "missing_packets": len(missing),
        "walk_stop_after_first_missing_s": response_s,
        "healthy_max_interval_s": max((v["max_capture_interval_s"] for key in
            ("find_source_health", "walk_source_health", "final_source_health")
            if (v := summary[key]) and v["authorized"]), default=None),
        "media_aligned": True}


def check_office_challenge() -> dict:
    root = RUNS / "office_3"
    old = root / "depth2m_baseline"
    current = root / "none"
    old_summary = json.loads((old / "summary.json").read_text())
    summary = json.loads((current / "summary.json").read_text())
    old_find = json.loads((old / "find_present/trace.json").read_text())
    new_find = json.loads((current / "find_present/trace.json").read_text())
    assert old_summary["present_action"] == "report_unidentified"
    assert old_find["target_visible_in_any_view"]
    assert old_find["verification"]["selected"]["local_depth"][
        "status"] == "insufficient_depth"
    assert old_find["verification"]["evaluator_selected_target_iou"] >= .25
    assert sha(old / "find_present/rgb_000.png") == sha(
        current / "find_present/rgb_000.png")
    old_depth = np.asarray(Image.open(old / "find_present/depth_003.png"))
    selected = old_find["verification"]["selected"]["box"]
    x = round((selected[0] + selected[2]) / 2)
    y = round((selected[1] + selected[3]) / 2)
    patch = old_depth[y-2:y+3, x-2:x+3].astype(float) / 1000.
    assert 2. < np.median(patch) < 4. and np.ptp(patch) < .05
    assert new_find["metric_candidate_max_depth_m"] == 4.
    assert summary["present_action"] == "candidate_to_owner_for_confirmation"
    assert summary["absent_action"] == "report_unidentified"
    assert summary["scripted_owner_yes"]
    assert summary["walk_result"] == "forward_range_blocked"
    assert summary["walk_accepted_forward_steps"] > 0
    assert summary["final_action"] is None
    assert summary["find_source_health"]["authorized"] and summary[
        "walk_source_health"]["authorized"]
    stages = {name: check_visual_stage(current / name) for name in
              ("find_present", "find_absent")}
    walk = json.loads((current / "walk/trace.json").read_text())
    assert video_frames(current / "walk/first_person.mp4") == len(walk["actions"])
    assert walk["actions"][-1]["decision"] == "hold"
    assert walk["actions"][-1]["reason"] == "forward_range_blocked"
    return {"room": "office_3", "old_depth_limit_m": 2.,
        "observed_verified_depth_patch_m": float(np.median(patch)),
        "new_depth_limit_m": 4.,
        "present_identity_after_depth_extension": summary["present_action"],
        "absent_action": summary["absent_action"],
        "bounded_forward_steps": summary["walk_accepted_forward_steps"],
        "route_result": summary["walk_result"],
        "final_target_gap_m": summary["evaluator_final_target_gap_m"],
        "source_health_through_walk": True,
        "visual_stage_media_aligned": stages,
        "next_task": "Check an observed-free bypass, then test route recovery under pose faults and moving entrants; do not lower the range clearance."}


def check_office_route_recovery() -> dict:
    root = RUNS / "office_3"
    old_walk = json.loads((root / "none/walk/trace.json").read_text())
    folder = root / "route_recovery"
    trace = json.loads((folder / "trace.json").read_text())
    source = json.loads((folder / "independent_range/packets.json").read_text())
    probe = json.loads((ROOT / "artifacts/ape-replica-interactive-20261015/office_route_recovery_probe.json").read_text())
    actions = trace["actions"]
    packets = source["packets"]
    assert trace["result"] == "observed_target_standoff"
    assert actions[0]["pose_xz_m"] == old_walk["actions"][-1]["actor_xz_m"]
    assert trace["observed_target_xz_m"] == old_walk["actions"][-1][
        "observed_target_xz_m"]
    assert video_frames(folder / "first_person.mp4") == len(actions)
    assert all(a["source_guard"]["authorized"] and
               a["source_guard"]["last_packet_age_s"] <= .2 and
               a["nearest_range_m"] >= trace["route_clearance_m"]
               for a in actions)
    steps = [a for a in actions if a["route"]["intent"] == "step_to_waypoint"]
    turns = [a for a in actions if a["route"]["intent"] == "turn_to_waypoint"]
    assert steps and all(a.get("motion_accepted") and
                         0 < a["route"]["forward_step_m"] <= .08 for a in steps)
    assert turns and all(abs(a["route"]["yaw_delta_deg"]) <= 10. for a in turns)
    assert trace["source_health"]["authorized"] and trace[
        "final_source_health"]["authorized"]
    assert len(packets) == trace["range_packets"]
    assert [p["index"] for p in packets] == list(range(len(packets)))
    assert all(p["source_health"] == "healthy" for p in packets)
    assert max(b["capture_wall_s"] - a["capture_wall_s"] for a, b in
               zip(packets, packets[1:])) <= .2
    final = check_visual_stage(folder / "final_view")
    assert final["action"] == trace["effective_final_action"] == \
        "candidate_to_owner_for_confirmation"
    assert final["selected_target_iou"] >= .25
    assert probe["plans"][0]["intent"] in ("turn_to_waypoint", "step_to_waypoint")
    return {"route_result": trace["result"], "route_frames": len(actions),
        "accepted_steps": len(steps), "turns": len(turns),
        "evaluator_final_target_gap_m": trace["evaluator_final_target_gap_m"],
        "final_visual_action": final["action"],
        "final_selected_target_iou": final["selected_target_iou"],
        "source_packets": len(packets), "media_aligned": True}


def main() -> None:
    rows = []
    for room in ROOMS:
        arms = {arm: check_arm(room, arm) for arm in ARMS}
        baseline = RUNS / room / "none"
        for arm in ("walk", "final"):
            other = RUNS / room / arm
            for name in ("owner/owner_rgb.png", "teach/teach_rgb.png"):
                assert sha(baseline / name) == sha(other / name)
            for stage in ("find_present", "find_absent"):
                frames = arms["none"]["stages"][stage]["frames"]
                assert frames == arms[arm]["stages"][stage]["frames"]
                assert all(sha(baseline / stage / f"rgb_{i:03d}.png") ==
                           sha(other / stage / f"rgb_{i:03d}.png")
                           for i in range(frames))
        assert all(sha(baseline / "final_view" / f"rgb_{i:03d}.png") ==
                   sha(RUNS / room / "final/final_view" / f"rgb_{i:03d}.png")
                   for i in range(check_visual_stage(baseline / "final_view")["frames"]))
        assert arms["none"]["accepted_forward_steps"] == arms[
            "final"]["accepted_forward_steps"]
        rows.append({"room": room, "arms": {arm: {
            key: value for key, value in data.items() if key != "scenario"}
            for arm, data in arms.items()},
            "paired_owner_teach_find_rgb_identical": True,
            "paired_healthy_final_fault_rgb_identical": True})
    office = check_office_challenge()
    recovery = check_office_route_recovery()
    report = {"schema": "ape.replica-owner-full-monitor-audit.v1",
        "room_count": len(rows) + 1, "paired_room_count": len(rows),
        "paired_arms_per_room": list(ARMS),
        "rows": rows,
        "office_route_challenge": office,
        "office_route_recovery": recovery,
        "conclusion": "The same visual task evidence yields final offers in two healthy rooms, stops after three accepted steps when the independent range source drops during walking, and withholds otherwise identical final visual offers when it drops during final inference. A separate observed-free route continuation recovers the blocked office approach and reidentifies the item.",
        "limits": "Three development-exposed static rooms, paired synthetic one-second source gaps in two rooms, static-room-only ideal range, scripted owner, command pose and 17cm disk. Office recovery is a separate development continuation, not an integrated frozen-policy transfer or capture-disjoint result. This checks software gating and packet timing, not moving-person clearance, physical stop response, sensor placement, IMU/joint/contact choice, or learned-policy transfer."}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"room_count": len(rows) + 1, "rooms": [x["room"] for x in rows] + [office["room"]],
        "walk_stop_after_missing_s": [x["arms"]["walk"][
            "walk_stop_after_first_missing_s"] for x in rows]}, indent=2))


if __name__ == "__main__":
    main()
