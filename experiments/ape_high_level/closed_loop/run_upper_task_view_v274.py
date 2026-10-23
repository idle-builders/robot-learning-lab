"""V274: guarded upper-view recovery when lower floor-view has no candidate."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_direct_structural_entry_v271 import (OUT as V271_OUT, V149, V150)
from direct_structural_rooms_v270 import DirectStructuralRoom, selected_visual
from fast_range_guard import FastRangeGuard
from run import ROOT, Vision, iou, sensed_range, sha
from run_color_grounded_entry_v151 import infer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v274"
V271 = V271_OUT / "direct-structural-entry-audit.json"
CASES = ((130, "raised", "present"),
         (130, "raised", "absent"),
         (130, "raised", "identical_lookalike"),
         (129, "screen", "present"))
ANGLES = (-.6, -1.2, 0., .6, 1.2)


def evaluate(seed: int, kind: str, mode: str, vision: Vision,
             exemplar: np.ndarray, owner_hist: np.ndarray,
             lower_entry: dict) -> dict:
    if any(view["accepted_box"] is not None for view in lower_entry["rows"]):
        raise ValueError("Upper recovery requires no lower-view candidate")
    room = DirectStructuralRoom(kind=kind, mode=mode,
                                side=1 if seed % 2 else -1)
    room.layout(seed, absent=mode != "present", furnished=True)
    branch = OUT / f"seed{seed}-{kind}-{mode}"
    branch.mkdir(parents=True, exist_ok=True)
    guards = {height: FastRangeGuard(band_m=.075,
              near_cluster_min_points=2,
              single_point_critical_band_m=.07) for height in (.04, .11)}
    serial = 0

    def scan(height: float):
        nonlocal serial
        serial += 1
        return sensed_range(room, seed, 274000 + serial, height_m=height)

    for height, guard in guards.items():
        guard.update(scan(height), room.estimated, room.time_s)

    def probe(substep: int, fraction: float):
        readings = {height: guard.update(
            scan(height), room.estimated, room.time_s)
                    for height, guard in guards.items()}
        hazard = ("near_body_change", "near_body_cluster", "critical_body_point")
        if any(row["status"] in hazard for row in readings.values()):
            return {"substep": substep, "fraction": fraction,
                    "readings": readings}
        return None

    rows = []
    accepted = []
    try:
        with imageio.get_writer(branch / "upper-task-sweep.mp4", fps=3,
                                codec="libx264", quality=7) as video:
            for yaw in ANGLES:
                room.look(yaw, "lower_rgb", safety_probe=probe)
                alarm = room.safety_stop_observation
                rgb, depth = room.capture("upper_rgb", depth=True)
                limited = depth.copy()
                limited[(limited < .08) | (limited > 1.5)] = np.nan
                camera = room.camera_pose_estimate("upper_rgb")
                choice = infer(vision, exemplar, owner_hist, rgb, limited,
                               camera, None, None)
                truth = room.truth_box("TomatoSauce", "upper_rgb")
                correct = (choice["accepted_box"] is not None and
                           truth is not None and
                           iou(choice["accepted_box"], truth) >= .25)
                frame = branch / f"upper-yaw-{yaw:+.1f}.npz"
                np.savez_compressed(frame, rgb=np.asarray(rgb), depth_m=limited,
                                    camera_origin_m=camera[0],
                                    camera_rotation=camera[1],
                                    actor_qpos=room.pose_data.qpos.copy(),
                                    estimated_pose_xyyaw=room.estimated.copy(),
                                    capture_time_s=room.time_s)
                panel = Image.fromarray(np.concatenate(
                    (np.asarray(rgb), np.asarray(room.capture_external())),
                    axis=1))
                draw = ImageDraw.Draw(panel)
                if choice["accepted_box"] is not None:
                    draw.rectangle(choice["accepted_box"],
                                   outline=(255, 215, 0), width=3)
                draw.rectangle((0, 0, panel.width, 29), fill=(0, 0, 0))
                draw.text((8, 6), f"Direct Frame upper task view | {kind} | {mode} | yaw{yaw:+.1f}",
                          fill=(255, 255, 255))
                video.append_data(np.asarray(panel))
                row = {"yaw_rad": yaw, "capture_time_s": room.time_s,
                       "actor_input_sha256": sha(frame),
                       "accepted_box": choice["accepted_box"],
                       "accepted_similarity": choice["accepted_score"],
                       "accepted_hue_intersection": choice["accepted_hue_intersection"],
                       "accepted_world_xyz_m": choice["accepted_world_xyz_m"],
                       "evaluator_truth_box": truth,
                       "evaluator_correct_item": correct,
                       "range_alarm": alarm}
                rows.append(row)
                if choice["accepted_world_xyz_m"] is not None:
                    accepted.append(row)
                if alarm is not None:
                    break
        xyz = np.asarray([row["accepted_world_xyz_m"] for row in accepted])
        if len(xyz) >= 2:
            median = np.median(xyz, axis=0)
            spread = max(float(np.linalg.norm(p[:2] - median[:2])) for p in xyz)
        else:
            median, spread = None, None
        raised = (median is not None and spread <= .08 and
                  median[2] >= .2 and all(row["range_alarm"] is None
                                              for row in rows))
        # For an exact taught *instance*, visually identical packaging does
        # not supply identity. The scripted owner reply is evaluator input.
        owner_yes = bool(raised and mode == "present")
        owner_question = bool(raised)
        status = ("owner_confirmed_raised_visual_only" if owner_yes else
                  "owner_rejected_lookalike" if owner_question else
                  "no_upper_candidate_unknown")
        return {"seed": seed, "kind": kind, "mode": mode,
                "lower_candidate_count": 0, "upper_accepted_view_count": len(accepted),
                "upper_world_median_xyz_m": None if median is None else median.tolist(),
                "upper_world_xy_spread_m": spread,
                "raised_support_evidence": bool(raised),
                "owner_questions": int(owner_question),
                "scripted_owner_yes_evaluator_only": owner_yes,
                "status": status, "base_motion_requested": False,
                "evaluator_min_model_clearance_m":
                    round(room.path_min_clearance[0], 4),
                "rows": rows,
                "video_sha256": sha(branch / "upper-task-sweep.mp4")}
    finally:
        room.close()


def main() -> None:
    entry = json.loads(V271.read_text())
    if entry.get("partial"):
        raise ValueError("V271 incomplete")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for seed, kind, mode in CASES:
        lower = next(row["entry"] for row in entry["rows"]
                     if row["seed"] == seed and row["mode"] == mode)
        row = evaluate(seed, kind, mode, vision, exemplar, owner_hist, lower)
        rows.append(row)
        print(seed, kind, mode, row["status"],
              "upper", row["upper_accepted_view_count"], flush=True)
    result = {"scope": "V274 actor-only active upper-head view after V271 lower-head sweep had zero candidates. Same frozen owner-taught DINO/hue thresholds, 5 guarded head-yaw views, ideal co-aligned 8-150cm upper depth hypothesis, two view XY-consistency<=8cm and z>=20cm for raised support; no base motion or floor route. Evaluator target boxes only score; scripted owner yes/no after supported visual candidate supplies exact-instance identity. Upper depth package/extrinsics/noise remain hypothetical, real owner voice not implemented. Screen scene remains no candidate, requiring a separate information-viewpoint action.",
              "source_sha256": sha(SOURCE), "v271_sha256": sha(V271),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "upper-task-view-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
