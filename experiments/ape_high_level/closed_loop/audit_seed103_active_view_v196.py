"""V196: stationary selected-head yaw screen at V195 seed-103 stop."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_side_look_target_v81 import infer
from run import ROOT, Room, Vision, iou, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v196"
V195_TRACE = ROOT / "artifacts/ape-closed-loop-cove50-v195/scene103/seed103/trace.json"
V195_FRAME = ROOT / "artifacts/ape-closed-loop-cove50-v195/scene103/seed103/frame-11.npz"
V192 = ROOT / "artifacts/ape-closed-loop-cove50-v192/third-split-geometry-audit.json"
EXEMPLAR = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
ANGLES = tuple(round(float(x), 1) for x in np.arange(0., 1.40001, .1))


def prepare(room: Room, *, absent: bool, positions: dict,
            actual: np.ndarray, estimated: np.ndarray,
            qpos: np.ndarray) -> None:
    room.reset()
    room.layout(103, absent=absent, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = estimated
    room._forward()


def capture(room: Room, yaw: float, memory: np.ndarray,
            vision: Vision, exemplar: np.ndarray,
            label: str) -> tuple[dict, Image.Image]:
    room.head = yaw
    room._forward()
    rgb, depth = room.capture("lower_rgb", depth=True)
    camera = room.camera_pose_estimate("lower_rgb")
    truth = room.truth_box("TomatoSauce", "lower_rgb") if label == "present" else None
    path = OUT / f"{label}-yaw{yaw:.1f}"
    raw_path = OUT / f"{label}-yaw{yaw:.1f}.npz"
    choice = infer(vision, exemplar, rgb, depth, camera, memory, None,
                   path.with_name(path.name + "-annotated.png"))
    np.savez_compressed(raw_path, rgb=np.asarray(rgb),
                        depth_m=depth, camera_origin_m=camera[0],
                        camera_rotation=camera[1],
                        actor_qpos=room.pose_data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy())
    scored = []
    for proposal in choice["proposals"]:
        scored.append({**proposal,
                       "evaluator_truth_iou": round(iou(proposal["box"], truth), 4)})
    selected = choice["accepted_box"]
    correct = selected is not None and truth is not None and iou(selected, truth) >= .25
    gap, body, scene = room._current_obstacle_clearance()
    return {"yaw_rad": yaw, "label": label,
            "accepted_box": selected,
            "accepted_correct_evaluator_only": bool(correct),
            "evaluator_truth_box": truth,
            "evaluator_body_clearance_m": round(gap, 5),
            "evaluator_limiting_geoms": [body, scene],
            "proposals": scored,
            "raw_sha256": sha(raw_path)}, Image.open(
                path.with_name(path.name + "-annotated.png")).copy()


def main() -> None:
    trace = json.loads(V195_TRACE.read_text())
    assert trace["seed"] == 103 and trace["status"] == "information_goal_reached"
    scene = next(row for row in json.loads(V192.read_text())["rows"]
                 if row["seed"] == 103)
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    with np.load(V195_FRAME) as saved:
        qpos = saved["actor_qpos"].copy()
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
    estimated = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
    memory = np.asarray(trace["final_actor_target_xy_m"], dtype=float)
    variant = {"CreamCheese": tuple(positions["CreamCheese"]),
               "Parmesan": tuple(positions["Parmesan"])}
    present = MovedPropRoom(variant)
    absent = MovedPropRoom(variant)
    vision = Vision("generic")
    exemplar = np.load(EXEMPLAR)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        prepare(present, absent=False, positions=positions,
                actual=actual, estimated=estimated, qpos=qpos)
        prepare(absent, absent=True, positions=positions,
                actual=actual, estimated=estimated, qpos=qpos)
        with imageio.get_writer(OUT / "paired-head-sweep.mp4", fps=4,
                                codec="libx264", quality=7) as video:
            for yaw in ANGLES:
                a, aimg = capture(present, yaw, memory, vision, exemplar, "present")
                b, bimg = capture(absent, yaw, memory, vision, exemplar, "absent")
                rows.extend((a, b))
                panel = Image.fromarray(np.concatenate((np.asarray(aimg),
                                                        np.asarray(bimg)), axis=1))
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 26), fill=(0, 0, 0))
                draw.text((8, 6), f"seed103 stationary yaw {yaw:.1f} | present accepted {a['accepted_box'] is not None}", fill=(255, 255, 255))
                draw.text((aimg.width + 8, 6),
                          f"matched absent accepted {b['accepted_box'] is not None}",
                          fill=(255, 255, 255))
                video.append_data(np.asarray(panel))
                print(yaw, "present", a["accepted_box"] is not None,
                      "correct", a["accepted_correct_evaluator_only"],
                      "absent", b["accepted_box"] is not None, flush=True)
    finally:
        present.close()
        absent.close()
    present_rows = [row for row in rows if row["label"] == "present"]
    absent_rows = [row for row in rows if row["label"] == "absent"]
    result = {"scope": "V196 revealed seed103 stationary selected lower-head yaw0.0-1.4rad/0.1rad, exact V195 final base and joints, same V81 DINO>=0.35/memory<=12cm actor gate, matched absent, evaluator IoU/clearance after actor choice. Read-only optical screen, no turn motion or route authorization.",
              "source_sha256": sha(SOURCE),
              "v195_trace_sha256": sha(V195_TRACE),
              "v195_frame_sha256": sha(V195_FRAME),
              "v192_sha256": sha(V192),
              "exemplar_sha256": sha(EXEMPLAR),
              "correct_present_views": sum(row["accepted_correct_evaluator_only"]
                                           for row in present_rows),
              "wrong_present_views": sum(row["accepted_box"] is not None and
                                         not row["accepted_correct_evaluator_only"]
                                         for row in present_rows),
              "accepted_absent_views": sum(row["accepted_box"] is not None
                                           for row in absent_rows),
              "min_sampled_full_body_clearance_m": min(
                  row["evaluator_body_clearance_m"] for row in rows),
              "screen_pass": (sum(row["accepted_correct_evaluator_only"]
                                  for row in present_rows) >= 2 and
                              all(row["accepted_box"] is None for row in absent_rows)),
              "video_sha256": sha(OUT / "paired-head-sweep.mp4"),
              "rows": rows}
    (OUT / "seed103-head-view-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
