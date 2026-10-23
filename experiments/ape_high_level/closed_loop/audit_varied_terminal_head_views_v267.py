"""V267: can a safe yaw-only look recover the V265 terminal can view?"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from run import ROOT, Vision, iou, sha
from run_rgb_floor_target_v213 import RgbFloorTargetBank
from run_varied_dynamic_loop_v265 import VariedDynamicShortRoom, V149, V150, V262


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v267"
V265_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v265"
V265 = V265_ROOT / "varied-dynamic-live-audit.json"
HEAD_YAWS_RAD = (-1.2, -.9, -.6, -.3, 0., .3, .6, .9, 1.2)
MIN_CLEARANCE_M = .05


def saved_bank(seed: int, vision: Vision) -> tuple[list[np.ndarray], list[dict]]:
    branch = V265_ROOT / f"scene{seed}" / f"seed{seed}"
    events = json.loads((branch / "online-color-bank-events.json").read_text())
    crops = [event for event in events
             if event["source"] == "original_owner_accepted_crop"][-2:]
    if len(crops) != 2:
        raise AssertionError(f"Seed {seed} lacks two owner-approved crops")
    features, sources = [], []
    for event in crops:
        frame = branch / f"frame-{event['call_index']}.npz"
        with np.load(frame) as saved:
            rgb = Image.fromarray(saved["rgb"])
        feature = np.asarray(vision.feature([rgb.crop(event["crop_box"])])[0],
                             dtype=float)
        digest = hashlib.sha256(feature.tobytes()).hexdigest()
        if digest != event["feature_sha256"]:
            raise AssertionError(f"Seed {seed} recent feature mismatch")
        features.append(feature)
        sources.append({"frame_sha256": sha(frame),
                        "crop_box": event["crop_box"],
                        "feature_sha256": digest})
    return features, sources


def episode(seed: int, scene: dict, vision: Vision,
            owner: np.ndarray, owner_hist: np.ndarray) -> dict:
    branch = V265_ROOT / f"scene{seed}" / f"seed{seed}"
    trace = json.loads((branch / "trace.json").read_text())
    last = max(branch.glob("frame-*.npz"),
               key=lambda path: int(path.stem.split("-")[1]))
    with np.load(last) as saved:
        qpos = saved["actor_qpos"].copy()
        time_s = float(saved["capture_time_s"])
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
    estimated = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
    memory = np.asarray(trace["final_actor_target_xy_m"], dtype=float)
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    moved = {"CreamCheese": tuple(positions["CreamCheese"]),
             "Parmesan": tuple(positions["Parmesan"])}
    room = VariedDynamicShortRoom(moved)
    recent, sources = saved_bank(seed, vision)
    rows = []
    video_path = OUT / f"seed{seed}-head-yaw-screen.mp4"
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room.data.qpos[:] = qpos
        room.actual[:] = actual
        room.estimated[:] = estimated
        original_head = float(qpos[room.head_addr])
        room.head = original_head
        room.time_s = time_s
        room._update_moving_obstacle()
        room._forward()
        with imageio.get_writer(video_path, fps=3, codec="libx264",
                                quality=7) as video:
            for yaw in HEAD_YAWS_RAD:
                sampled_clearance = float("inf")
                for head in np.linspace(original_head, yaw, 21):
                    room.head = float(head)
                    room._forward()
                    sampled_clearance = min(sampled_clearance,
                        room._current_obstacle_clearance()[0])
                room.head = yaw
                room._forward()
                rgb, _ = room.capture("lower_rgb", depth=False)
                camera = room.camera_pose_estimate("lower_rgb")
                truth = room.truth_box("TomatoSauce", "lower_rgb")
                bank = RgbFloorTargetBank(owner, owner_hist)
                bank.recent = [feature.copy() for feature in recent]
                annotated_path = OUT / f"seed{seed}" / f"yaw{yaw:+.1f}.png"
                annotated_path.parent.mkdir(parents=True, exist_ok=True)
                detection = bank.infer(vision, owner, rgb, None, camera,
                                       memory, truth, annotated_path)
                chosen = next((r for r in detection["proposals"]
                               if r["box"] == detection["accepted_box"]), None)
                correct = (chosen is not None and truth is not None and
                           iou(chosen["box"], truth) >= .25)
                safe = sampled_clearance >= MIN_CLEARANCE_M
                row = {"seed": seed, "yaw_rad": yaw,
                       "sampled_head_arc_min_clearance_m": round(
                           sampled_clearance, 4),
                       "safe_5cm_evaluator_only": safe,
                       "evaluator_truth_box": truth,
                       "proposal_count": len(detection["proposals"]),
                       "proposals": [{"box": proposal["box"],
                                      "similarity": proposal["similarity"],
                                      "owner_similarity": proposal["owner_similarity"],
                                      "owner_hue_intersection": proposal[
                                          "owner_hue_intersection"],
                                      "distance_to_memory_m": proposal[
                                          "distance_to_memory_m"],
                                      "actor_gate_accept": proposal[
                                          "actor_gate_accept"],
                                      "evaluator_target_iou": proposal[
                                          "evaluator_target_iou"]}
                                     for proposal in detection["proposals"]],
                       "accepted_box": detection["accepted_box"],
                       "accepted_correct_evaluator_only": correct,
                       "accepted_similarity": detection["accepted_similarity"],
                       "accepted_hue_intersection": None if chosen is None else
                           chosen["owner_hue_intersection"],
                       "accepted_memory_distance_m": None if chosen is None else
                           chosen["distance_to_memory_m"],
                       "annotated_sha256": sha(annotated_path)}
                rows.append(row)
                panel = Image.fromarray(np.concatenate(
                    (np.asarray(rgb), np.asarray(room.capture_external())),
                    axis=1))
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 32), fill=(0, 0, 0))
                draw.text((8, 9),
                          f"seed {seed} yaw {yaw:+.1f} | accepted {chosen is not None} "
                          f"correct {correct} | clearance {sampled_clearance:.3f}m",
                          fill=(255, 255, 255))
                video.append_data(np.asarray(panel))
    finally:
        room.close()
    return {"seed": seed, "v265_stop_status": trace["status"],
            "v265_last_frame_sha256": sha(last), "bank_sources": sources,
            "saved_stop_time_s": time_s,
            "saved_head_yaw_rad": original_head,
            "safe_correct_yaw_count": sum(r["safe_5cm_evaluator_only"] and
                                          r["accepted_correct_evaluator_only"]
                                          for r in rows),
            "rows": rows, "video_sha256": sha(video_path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = {s["seed"]: s for s in json.loads(V262.read_text())["rows"]}
    baseline = json.loads(V265.read_text())
    seeds = [row["seed"] for row in baseline["rows"] if row.get("moves", 0) > 0]
    if seeds != [115, 117, 121, 125]:
        raise AssertionError(f"Unexpected V265 stop set: {seeds}")
    vision = Vision("generic")
    owner = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for seed in seeds:
        row = episode(seed, scenes[seed], vision, owner, owner_hist)
        rows.append(row)
        print(seed, "safe correct yaws", row["safe_correct_yaw_count"],
              "/", len(HEAD_YAWS_RAD), flush=True)
    report = {"scope": "V267 evaluator-only yaw-only optical feasibility screen from all four V265 selected-Cove50 moving-room safe stop poses. Restore exact saved actual/estimated pose, joint state, furniture, static final stool and last two hashed owner-approved appearance prototypes. Nine lower-head yaw poses; frozen V201 DINO>=.35, owner hue>=.70 and RGB floor-ray <=12cm memory gate. Truth box/IoU and sampled 21-pose head-arc body clearance are evaluator labels only, never used to choose a yaw or motion. No live actor recovery, owner question, new base motion, timed range guard, same-red absent challenge or physical head qualification.",
              "source_sha256": sha(SOURCE), "v262_sha256": sha(V262),
              "v265_sha256": sha(V265), "v149_sha256": sha(V149),
              "v150_sha256": sha(V150),
              "yaw_radians": HEAD_YAWS_RAD,
              "min_evaluator_clearance_m": MIN_CLEARANCE_M,
              "rows": rows}
    (OUT / "varied-terminal-head-view-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
