"""Independent packet, geometry, fresh-view and video audit of V599."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance
import audit_robocasa_person_joined_v498 as base_audit
import robocasa_person_interaction as interaction
import run_robocasa_memory_return_v599 as trial
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion


OUT = trial.OUT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        return sum(1 for frame in reader if frame.ndim == 3 and
                   frame.shape[2] == 3)


def audit_episode(ep: dict) -> dict:
    summary = ep["memory_return"]
    if summary is None:
        raise AssertionError("Owner turn did not reach memory continuation")
    case = ep["scenario"]
    model = joined.scene_with_owner_and_prop(case["from_xy_m"])
    fingerprint = hashlib.sha256(b"".join(array.tobytes() for array in
        (model.body_pos, model.geom_pos, model.geom_quat,
         model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    assert fingerprint == ep["scene_fingerprint"]
    held_rows = [row for row in ep["trace"] if row["stage"] == "safe_side_hold"]
    pre_memory = {**ep,
                  "final_actual_xy_m": ep["route_episode"]["end_true_xy_m"],
                  "hold_until_s": (held_rows[-1]["time_s"] if held_rows else
                                   ep["route_episode"]["end_time_s"])}
    baseline = base_audit.audit_episode(pre_memory)
    assert baseline["scene_fingerprint_pass"]
    assert baseline["ape_5cm_margin_pass"]
    assert baseline["person_room_5cm_pass"]
    assert baseline["can_room_nonoverlap_pass"]
    assert baseline["video_sha_pass"] and baseline["video_frames_pass"]
    data = mujoco.MjData(model)
    interaction.show_pose(model, 1.)
    prop_body = model.body("route_obstacle").id
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    script = motion(case)
    minimum = .1
    movement_samples = 0
    decisions = []
    fresh = []
    removed = False
    for row in ep["trace"]:
        if row["stage"] == "owner_removes_item_evaluator_script":
            model.body_pos[model.body("shown_can").id] = (-10., -10., .43)
            removed = True
        if row["stage"] == "memory_fresh_view":
            fresh.append(row)
            assert sha(Path(row["image_path"])) == row["image_sha256"]
            assert sha(Path(row["depth_path"])) == row["depth_sha256"]
        if row["stage"] != "memory_route_decision":
            continue
        decisions.append(row)
        command = row.get("actor_bounded_command_xy_m")
        if command is not None:
            assert np.linalg.norm(command) <= .10001
        assert row["range_capture_time_s"] <= row["time_s"] + .001
        assert len(row["range_72_beams_m"]) == 72
        for sub in row["motion_substeps"]:
            if sub["stopped_before_move"]:
                continue
            xy = np.asarray(sub["evaluator_actual_xy_m"])
            model.body_pos[prop_body, :2] = script(float(sub["time_s"]))
            gap, _ = clearance(model, data, ape_body, ape_geom, xy)
            minimum = min(minimum, gap)
            assert abs(gap - sub["evaluator_proxy_gap_m"]) <= .00002
            movement_samples += 1
    assert round(minimum, 5) == summary["minimum_proxy_gap_m_evaluator_only"]
    assert len(decisions) >= summary["moves"]
    assert len(fresh) >= 2 if summary["route_status"] == "running" else not fresh
    assert removed == (summary["item_state"] == "removed_while_away")
    video = Path(ep["video"]["path"])
    assert sha(video) == ep["video"]["sha256"]
    count = frames(video)
    assert count == ep["video"]["frames"]
    wrong_claim = False
    if summary["item_status"] == "fresh_metric_item_refound":
        assert summary["item_state"] == "retained"
        assert summary["source"] in ("range", "rgbd")
        assert summary["last_pose_health"] == "registered"
        assert summary["fresh_track_status"] == "visually_reacquired_unlocalized"
        assert summary["fresh_frame_age_s"] <= interaction.owner.MAX_FRESH_AGE_S
        assert fresh[-1]["item_box_iou_evaluator_only"] >= .5
        assert summary["fresh_item_error_m_evaluator_only"] < .1
    if summary["item_state"] == "removed_while_away":
        wrong_claim = summary["item_status"] == "fresh_metric_item_refound"
    return {"source": summary["source"], "item_state": summary["item_state"],
            "owner_outcome": ep["outcome"],
            "route_status": summary["route_status"],
            "item_status": summary["item_status"],
            "moves": summary["moves"],
            "pose_error_m_evaluator_only": summary["end_pose_error_m_evaluator_only"],
            "minimum_proxy_gap_m_evaluator_only": round(minimum, 5),
            "pre_memory_minimum_proxy_gap_m_evaluator_only":
                baseline["nearest_ape_any"]["signed_gap_m_capped_at_10cm"],
            "movement_samples_20ms": movement_samples,
            "fresh_views": len(fresh),
            "video_frames": count,
            "wrong_claim": wrong_claim}


def main() -> None:
    declaration_path = OUT / "predeclared-memory-return.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == sha(declaration_path)
    assert declaration["source_sha256"] == sha(OUT / trial.SOURCE.name)
    assert declaration["joined_source_sha256"] == sha(
        OUT / Path(joined.__file__).name)
    assert tuple(declaration["sources"]) == trial.SOURCES
    assert tuple(declaration["item_states"]) == trial.ITEM_STATES
    assert len(report["episodes"]) == 6
    rows = [audit_episode(ep) for ep in report["episodes"]]
    assert sum(row["wrong_claim"] for row in rows) == 0
    result = {"schema": "robot-learning-lab.robocasa-memory-return-audit.v1",
              "report_sha256": sha(report_path),
              "declaration_sha256": sha(declaration_path),
              "videos_decoded": len(rows),
              "wrong_metric_item_claims": 0,
              "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "all_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
