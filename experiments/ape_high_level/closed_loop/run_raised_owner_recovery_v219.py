"""V219: guarded upper-view search, scripted owner check, fresh visual recheck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_entry_color_v150 import descriptor
from fast_range_guard import FastRangeGuard
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, sensed_range, sha
from run_raised_target_v217 import RaisedCanRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v219"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V217 = ROOT / "artifacts/ape-closed-loop-cove50-v217/raised-target-live-audit.json"
V218 = ROOT / "artifacts/ape-closed-loop-cove50-v218/raised-upper-search-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
ANGLES = (-1.2, -.6, 0., .6, 1.2)


def choose(vision: Vision, exemplar: np.ndarray, owner_hist: np.ndarray,
           rgb: Image.Image) -> tuple[dict | None, list[dict]]:
    proposals = vision.detect(rgb)
    features = (vision.feature([rgb.crop(p["box"]) for p in proposals])
                if proposals else np.zeros((0, len(exemplar))))
    rows = []
    for proposal, feature in zip(proposals, features):
        score = float(feature @ exemplar)
        color = descriptor(rgb, proposal["box"])
        overlap = float(np.minimum(owner_hist,
            np.asarray(color["hue_histogram"])).sum())
        rows.append({"box": proposal["box"],
                     "owner_similarity": round(score, 4),
                     "owner_hue_intersection": round(overlap, 4),
                     "candidate": score >= .35 and overlap >= .70})
    selected = max((r for r in rows if r["candidate"]),
                   key=lambda r: r["owner_similarity"], default=None)
    return selected, rows


def episode(seed: int, absent: bool, vision: Vision,
            exemplar: np.ndarray, owner_hist: np.ndarray) -> dict:
    scene = next(s for s in json.loads(V203.read_text())["rows"] if s["seed"] == seed)
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    moved = {"CreamCheese": tuple(positions["CreamCheese"]),
             "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"]
                                if absent else positions["Parmesan"])}
    room = RaisedCanRoom(moved)
    robot = RobotGeometry(room.model)
    trace = json.loads((ROOT / f"artifacts/ape-closed-loop-cove50-v217/scene{seed}/seed{seed}/trace.json").read_text())
    last = trace["events"][-1]
    frames = sorted((ROOT / f"artifacts/ape-closed-loop-cove50-v217/scene{seed}/seed{seed}").glob("frame-*.npz"),
                    key=lambda path: int(path.stem.split("-")[1]))
    branch = OUT / f"seed{seed}-{'absent' if absent else 'present'}"
    branch.mkdir(parents=True, exist_ok=True)
    with np.load(frames[-1]) as z:
        qpos = z["actor_qpos"].copy()
    monitor4 = FastRangeGuard(band_m=.075, near_cluster_min_points=2,
                              single_point_critical_band_m=.07)
    monitor11 = FastRangeGuard(band_m=.075, near_cluster_min_points=2,
                               single_point_critical_band_m=.07)
    serial = 0
    events = []
    question_count = 0
    owner_yes = False
    verified = False
    alarm = None
    candidate = None

    def scan(height: float):
        nonlocal serial
        serial += 1
        return sensed_range(room, seed, 2190000 + serial, height_m=height)

    def point_gap(height: float):
        def gap(points_xy: np.ndarray) -> np.ndarray:
            robot.set_state(robot.pose_qpos(room.data.qpos, room.estimated))
            return robot.solid_distances(
                np.c_[points_xy, np.full(len(points_xy), height)],
                robot.primitives())
        return gap

    def probe(substep: int, fraction: float):
        low = monitor4.update(scan(.04), room.estimated, room.time_s,
                              point_gap=point_gap(.04))
        high = monitor11.update(scan(.11), room.estimated, room.time_s,
                                point_gap=point_gap(.11))
        hazard = ("near_body_change", "near_body_cluster", "critical_body_point")
        if low["status"] in hazard or high["status"] in hazard:
            return {"low": low, "high": high, "substep": substep,
                    "fraction": fraction, "time_s": room.time_s}
        return None

    def panel(video, rgb: Image.Image, action: str, selected: dict | None):
        annotated = rgb.copy()
        if selected is not None:
            ImageDraw.Draw(annotated).rectangle(selected["box"],
                                                 outline=(0, 255, 0), width=3)
        combined = Image.fromarray(np.concatenate((np.asarray(annotated),
                    np.asarray(room.capture_external())), axis=1))
        draw = ImageDraw.Draw(combined)
        draw.rectangle((0, 0, combined.width, 34), fill=(0, 0, 0))
        draw.text((8, 8), f"V219 seed{seed} {'absent' if absent else 'present'} | {action} | t={room.time_s:.1f}s",
                  fill=(255, 255, 255))
        video.append_data(np.asarray(combined))

    try:
        room.reset()
        room.layout(seed, absent=absent, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room.data.qpos[:] = qpos
        room.actual[:] = last["evaluator_actual_pose_xyyaw"]
        room.estimated[:] = last["estimated_pose_xyyaw"]
        room.head = float(qpos[room.head_addr])
        room.time_s = float(last["time_s"])
        room._forward()
        monitor4.update(scan(.04), room.estimated, room.time_s,
                        point_gap=point_gap(.04))
        monitor11.update(scan(.11), room.estimated, room.time_s,
                         point_gap=point_gap(.11))
        with imageio.get_writer(branch / "owner-recovery.mp4", fps=3,
                                codec="libx264", quality=7) as video:
            for angle in ANGLES:
                room.look(angle, "lower_rgb", safety_probe=probe)
                alarm = room.safety_stop_observation
                if alarm is not None:
                    events.append({"action": "range_alarm_during_look", "alarm": alarm})
                    break
                rgb, _ = room.capture("upper_rgb", depth=False)
                candidate, proposals = choose(vision, exemplar, owner_hist, rgb)
                truth = room.truth_box("TomatoSauce", "upper_rgb")
                events.append({"action": "upper_identity_look", "yaw_rad": angle,
                               "time_s": round(room.time_s, 3),
                               "candidate": candidate, "proposals": proposals,
                               "evaluator_truth_box": truth,
                               "candidate_correct_evaluator_only":
                                   candidate is not None and iou(candidate["box"], truth) >= .25})
                panel(video, rgb, "upper look", candidate)
                if candidate is not None:
                    question_count += 1
                    # Scripted owner is an external participant. Truth enters
                    # only this simulated yes/no answer, after the actor asks.
                    owner_yes = truth is not None and iou(candidate["box"], truth) >= .25
                    events.append({"action": "owner_confirm_candidate",
                                   "question_count": question_count,
                                   "scripted_owner_yes_evaluator_only": owner_yes})
                    panel(video, rgb, "owner yes" if owner_yes else "owner no", candidate)
                    if owner_yes:
                        room.wait(safety_probe=probe)
                        alarm = room.safety_stop_observation
                        if alarm is not None:
                            events.append({"action": "range_alarm_during_wait", "alarm": alarm})
                            break
                        fresh, _ = room.capture("upper_rgb", depth=False)
                        second, second_proposals = choose(vision, exemplar, owner_hist, fresh)
                        verified = (second is not None and
                                    iou(second["box"], candidate["box"]) >= .6)
                        events.append({"action": "post_owner_fresh_recheck",
                                       "time_s": round(room.time_s, 3),
                                       "candidate": second,
                                       "same_image_region": verified,
                                       "proposals": second_proposals})
                        panel(video, fresh, "fresh recheck", second)
                    break
    finally:
        room.close()
    status = ("range_stop" if alarm is not None else
              "owner_confirmed_visual_find" if owner_yes and verified else
              "owner_rejected_candidate" if question_count and not owner_yes else
              "unverified_after_owner" if owner_yes else
              "no_upper_candidate_ask_help")
    result = {"seed": seed, "absent": absent, "v217_stop_status": trace["status"],
              "replay_last_frame_sha256": sha(frames[-1]),
              "status": status, "owner_questions": question_count,
              "scripted_owner_yes_evaluator_only": owner_yes,
              "fresh_visual_recheck": verified,
              "motion_request_after_recovery": None,
              "metric_target_pose_status": "unknown_support_and_range",
              "events": events,
              "video_sha256": sha(branch / "owner-recovery.mp4")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[107, 111, 113])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    rows = []
    for seed in args.seeds:
        for absent in (False, True):
            row = episode(seed, absent, vision, exemplar, owner_hist)
            rows.append(row)
            print(seed, "absent", absent, row["status"],
                  "questions", row["owner_questions"], flush=True)
    result = {"scope": "V219 revealed paired continuation from V217 raised-target memory-expiry stop, using existing upper RGB and five-angle kinematic head-yaw sweep, selected-body 10Hz virtual4/11cm guard on look/wait, frozen owner exemplar similarity>=.35 and hue>=.70 with old XY prior lifted only for an uncertain candidate. A question precedes scripted owner yes/no (truth only in external answer); a fresh post-answer RGB view must agree before visual-only found state. No metric location, target approach, physical speech/touch, native head skill or stop qualification. Paired absent puts Parmesan at old target XY. This is a bounded interaction recovery, not autonomous fetch or held-out validation.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v217_sha256": sha(V217), "v218_sha256": sha(V218),
              "rows": rows}
    (OUT / "owner-upper-recovery-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
