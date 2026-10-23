"""V249: guarded terminal owner clarification after the V247 safe stop."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_entry_color_v150 import descriptor
from audit_same_color_swap_v248 import recent_features, room_for_swap
from fast_range_guard import FastRangeGuard
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, locate_world, sensed_range, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v249"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V247_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v247"
V247 = V247_ROOT / "temporal-identity-valid-stool-live-audit.json"
V248 = ROOT / "artifacts/ape-closed-loop-cove50-v248/same-color-swap-audit.json"
QUESTION_YAW_RAD = .6
QUESTION_HUE_MIN = .67
QUESTION_BANK_SIM_MIN = .35
QUESTION_MEMORY_MAX_M = .12


def present_room(seed: int, qpos: np.ndarray, actual: np.ndarray,
                 estimated: np.ndarray) -> MovedPropRoom:
    scene = next(row for row in json.loads(V203.read_text())["rows"]
                 if row["seed"] == seed)
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                          "Parmesan": tuple(positions["Parmesan"])})
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
    room.set_table_height(.37)
    room.model.body_pos[room.model.body("loop_stool").id] = (.84, .35, .20)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = estimated
    room.head = float(qpos[room.head_addr])
    room.time_s = 28.
    room._forward()
    return room


def candidates(vision: Vision, owner: np.ndarray, recent: list[np.ndarray],
               owner_hist: np.ndarray, rgb: Image.Image,
               camera_pose: tuple[np.ndarray, np.ndarray],
               remembered_xy: np.ndarray) -> tuple[dict | None, list[dict]]:
    proposals = vision.detect(rgb)
    features = (vision.feature([rgb.crop(row["box"]) for row in proposals])
                if proposals else np.zeros((0, len(owner))))
    bank = np.stack([owner, *recent])
    rows = []
    for proposal, feature in zip(proposals, features):
        score = float(np.max(np.asarray(feature) @ bank.T))
        world = locate_world(proposal["box"], camera_pose, None)
        distance = (None if world is None else float(np.linalg.norm(
            np.asarray(world[:2]) - remembered_xy)))
        hue = float(np.minimum(owner_hist, np.asarray(
            descriptor(rgb, proposal["box"])["hue_histogram"])).sum())
        question_candidate = (score >= QUESTION_BANK_SIM_MIN and
                              distance is not None and
                              distance <= QUESTION_MEMORY_MAX_M and
                              hue >= QUESTION_HUE_MIN)
        rows.append({**proposal, "bank_similarity": round(score, 4),
                     "owner_hue_intersection": round(hue, 4),
                     "memory_distance_m": None if distance is None else
                        round(distance, 4),
                     "floor_world_xyz_m": world,
                     "question_candidate": question_candidate})
    selected = max((row for row in rows if row["question_candidate"]),
                   key=lambda row: row["bank_similarity"], default=None)
    return selected, rows


def episode(seed: int, case: str, vision: Vision,
            owner: np.ndarray, owner_hist: np.ndarray,
            recent: list[np.ndarray]) -> dict:
    branch = V247_ROOT / f"scene{seed}" / f"seed{seed}"
    trace = json.loads((branch / "trace.json").read_text())
    frame = max(branch.glob("frame-*.npz"),
                key=lambda path: int(path.stem.split("-")[1]))
    with np.load(frame) as saved:
        qpos = saved["actor_qpos"].copy()
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"])
    estimated = np.asarray(trace["final_estimated_pose_xyyaw"])
    memory = np.asarray(trace["final_actor_target_xy_m"])
    if case == "present":
        room = present_room(seed, qpos, actual, estimated)
    else:
        mode = {"ordinary_absent": "ordinary_absent",
                "red_lookalike_absent": "same_red_material"}[case]
        room = room_for_swap(seed, memory, mode, 1., qpos, actual, estimated)
    robot = RobotGeometry(room.model)
    guards = {height: FastRangeGuard(
        band_m=.075, near_cluster_min_points=2,
        single_point_critical_band_m=.07) for height in (.04, .11)}
    serial = 0
    events = []
    alarm = None
    selected = None
    owner_yes = False
    fresh = False
    episode_dir = OUT / f"seed{seed}-{case}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    def scan(height: float):
        nonlocal serial
        serial += 1
        return sensed_range(room, seed, 2490000 + serial, height_m=height)

    def point_gap(height: float):
        def gap(points_xy: np.ndarray) -> np.ndarray:
            robot.set_state(robot.pose_qpos(room.data.qpos, room.estimated))
            return robot.solid_distances(
                np.c_[points_xy, np.full(len(points_xy), height)],
                robot.primitives())
        return gap

    def probe(substep: int, fraction: float):
        observations = {height: guards[height].update(
            scan(height), room.estimated, room.time_s,
            point_gap=point_gap(height)) for height in guards}
        hazard = ("near_body_change", "near_body_cluster", "critical_body_point")
        if any(row["status"] in hazard for row in observations.values()):
            return {"observations": observations, "substep": substep,
                    "fraction": fraction, "time_s": room.time_s}
        return None

    def panel(video, rgb: Image.Image, caption: str, box: list[float] | None):
        left = rgb.copy()
        if box is not None:
            ImageDraw.Draw(left).rectangle(box, outline=(255, 215, 0), width=3)
        merged = Image.fromarray(np.concatenate(
            (np.asarray(left), np.asarray(room.capture_external())), axis=1))
        draw = ImageDraw.Draw(merged)
        draw.rectangle((0, 0, merged.width, 30), fill=(0, 0, 0))
        draw.text((8, 8), f"seed{seed} {case} | {caption} | t={room.time_s:.1f}s",
                  fill=(255, 255, 255))
        video.append_data(np.asarray(merged))

    video_path = episode_dir / "owner-clarification.mp4"
    try:
        for height in guards:
            guards[height].update(scan(height), room.estimated, room.time_s,
                                  point_gap=point_gap(height))
        with imageio.get_writer(video_path, fps=3, codec="libx264", quality=7) as video:
            room.look(QUESTION_YAW_RAD, "lower_rgb", safety_probe=probe)
            alarm = room.safety_stop_observation
            if alarm is not None:
                events.append({"action": "range_alarm_during_look", "alarm": alarm})
            else:
                rgb, _ = room.capture("lower_rgb", depth=True)
                camera = room.camera_pose_estimate("lower_rgb")
                selected, proposals = candidates(
                    vision, owner, recent, owner_hist, rgb, camera, memory)
                truth = room.truth_box("TomatoSauce", "lower_rgb")
                events.append({"action": "ambiguous_item_view",
                               "time_s": round(room.time_s, 3),
                               "candidate": selected, "proposals": proposals,
                               "evaluator_truth_box": truth})
                rgb.save(episode_dir / "question-view.png")
                panel(video, rgb, "candidate" if selected else "no candidate",
                      None if selected is None else selected["box"])
                if selected is not None:
                    # Evaluator truth becomes only the simulated owner's reply,
                    # after the actor has asked about an image candidate.
                    owner_yes = truth is not None and iou(selected["box"], truth) >= .25
                    events.append({"action": "owner_question",
                                   "scripted_owner_yes_evaluator_only": owner_yes})
                    panel(video, rgb, "owner yes" if owner_yes else "owner no",
                          selected["box"])
                    if owner_yes:
                        room.wait(safety_probe=probe)
                        alarm = room.safety_stop_observation
                        if alarm is not None:
                            events.append({"action": "range_alarm_during_wait",
                                           "alarm": alarm})
                        else:
                            second_rgb, _ = room.capture("lower_rgb", depth=True)
                            second_camera = room.camera_pose_estimate("lower_rgb")
                            second, second_rows = candidates(
                                vision, owner, recent, owner_hist,
                                second_rgb, second_camera, memory)
                            fresh = (second is not None and
                                     iou(second["box"], selected["box"]) >= .6)
                            events.append({"action": "fresh_post_answer_view",
                                           "time_s": round(room.time_s, 3),
                                           "candidate": second,
                                           "same_image_region": fresh,
                                           "proposals": second_rows})
                            panel(video, second_rgb,
                                  "confirmed fresh" if fresh else "recheck failed",
                                  None if second is None else second["box"])
    finally:
        room.close()
    status = ("range_stop" if alarm is not None else
              "owner_confirmed_visual_find" if owner_yes and fresh else
              "owner_rejected_candidate" if selected is not None and not owner_yes else
              "unverified_after_owner" if owner_yes else
              "no_candidate_ask_help")
    return {"seed": seed, "case": case, "v247_stop_status": trace["status"],
            "replay_last_frame_sha256": sha(frame), "status": status,
            "owner_questions": int(selected is not None),
            "scripted_owner_yes_evaluator_only": owner_yes,
            "fresh_post_answer_view": fresh,
            "motion_request_after_recovery": None,
            "metric_target_pose_status": "floor_ray_candidate_only",
            "events": events, "video_sha256": sha(video_path)}


def main() -> None:
    if not V248.exists():
        raise FileNotFoundError("Run V248 first")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    owner = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for seed in (111, 113):
        recent, _ = recent_features(seed, vision)
        for case in ("present", "ordinary_absent", "red_lookalike_absent"):
            row = episode(seed, case, vision, owner, owner_hist, recent)
            rows.append(row)
            print(seed, case, row["status"], "questions",
                  row["owner_questions"], flush=True)
    report = {"scope": "V249 revealed V247 final-pose guarded owner-clarification continuation. V248 actor bank and RGB floor candidate generator; question at lower-head yaw0.6 for image/memory/hue candidate, then scripted owner yes/no (evaluator truth confined to external reply), guarded wait and fresh same-region recheck before visual-only find. Paired present, ordinary absent and target-absent red-material different-mesh lookalike on seeds111/113. No base motion, metric support proof, real voice, native head/stop or physical sensor validation after replay. Exact-item identity comes from owner reply, not from color bank.",
              "source_sha256": sha(SOURCE), "v247_sha256": sha(V247),
              "v248_sha256": sha(V248), "rows": rows}
    (OUT / "terminal-owner-clarification-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
