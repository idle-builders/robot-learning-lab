"""V199: target-absent near-memory distractors against final online view bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_side_look_target_v81 import SIMILARITY_MIN, TARGET_DISTANCE_MAX_M
from run import ROOT, Vision, locate_world, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v199"
V198 = ROOT / "artifacts/ape-closed-loop-cove50-v198/online-view-bank-audit.json"
V198_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v198/scene103/seed103"
V192 = ROOT / "artifacts/ape-closed-loop-cove50-v192/third-split-geometry-audit.json"
OWNER = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
OFFSETS_M = (0., .04, .08)
ANGLES_RAD = tuple(round(float(v), 1) for v in np.arange(0., 1.40001, .2))


def score_view(room: MovedPropRoom, vision: Vision, bank: np.ndarray,
               memory: np.ndarray, yaw: float, case: str) -> tuple[dict, Image.Image]:
    room.head = yaw
    room._forward()
    rgb, depth = room.capture("lower_rgb", depth=True)
    raw_path = OUT / f"{case}-yaw{yaw:.1f}-rgb.png"
    rgb.save(raw_path)
    camera = room.camera_pose_estimate("lower_rgb")
    proposals = vision.detect(rgb)
    features = (vision.feature([rgb.crop(p["box"]) for p in proposals])
                if proposals else np.zeros((0, bank.shape[1])))
    rows = []
    for proposal, feature in zip(proposals, features):
        scores = np.asarray(feature) @ bank.T
        world = locate_world(proposal["box"], camera, depth)
        distance = None if world is None else float(np.linalg.norm(
            np.asarray(world[:2]) - memory))
        fixed = float(scores[0]) >= SIMILARITY_MIN and distance is not None and distance <= TARGET_DISTANCE_MAX_M
        joined = float(np.max(scores)) >= SIMILARITY_MIN and distance is not None and distance <= TARGET_DISTANCE_MAX_M
        rows.append({**proposal,
                     "similarities": [round(float(v), 4) for v in scores],
                     "distance_to_memory_m": None if distance is None else round(distance, 4),
                     "fixed_owner_pass": fixed,
                     "view_bank_pass": joined})
    fixed_choices = [row for row in rows if row["fixed_owner_pass"]]
    bank_choices = [row for row in rows if row["view_bank_pass"]]
    fixed_selected = max(fixed_choices, key=lambda row: row["similarities"][0]) if fixed_choices else None
    bank_selected = max(bank_choices, key=lambda row: max(row["similarities"])) if bank_choices else None
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    for row in rows:
        draw.rectangle(row["box"], outline=(255, 0, 0) if row is bank_selected else (255, 200, 0), width=3)
    path = OUT / f"{case}-yaw{yaw:.1f}-annotated.png"
    image.save(path)
    gap, body, obstacle = room._current_obstacle_clearance()
    return {"case": case, "yaw_rad": yaw,
            "fixed_owner_accepted_box": None if fixed_selected is None else fixed_selected["box"],
            "view_bank_accepted_box": None if bank_selected is None else bank_selected["box"],
            "evaluator_body_clearance_m": round(gap, 5),
            "evaluator_limiting_geoms": [body, obstacle],
            "proposals": rows,
            "raw_rgb_sha256": sha(raw_path),
            "annotated_sha256": sha(path)}, image


def main() -> None:
    result198 = json.loads(V198.read_text())
    assert result198["development_gate_pass"]
    trace = json.loads((V198_ROOT / "trace.json").read_text())
    frame_paths = sorted(V198_ROOT.glob("frame-*.npz"),
                         key=lambda path: int(path.stem.split("-")[-1]))
    with np.load(frame_paths[-1]) as saved:
        final_qpos = saved["actor_qpos"].copy()
    bank_events = [event for event in result198["bank_events"]
                   if event["source"] == "original_owner_accepted_crop"][-2:]
    assert len(bank_events) == 2
    vision = Vision("generic")
    owner = np.load(OWNER)
    bank = [owner]
    bank_sources = []
    for event in bank_events:
        path = V198_ROOT / f"frame-{event['call_index']}.npz"
        with np.load(path) as saved:
            image = Image.fromarray(saved["rgb"])
        feature = vision.feature([image.crop(event["crop_box"])])[0]
        bank.append(feature)
        bank_sources.append({"frame_sha256": sha(path),
                             "call_index": event["call_index"],
                             "crop_box": event["crop_box"],
                             "feature_sha256": hashlib.sha256(
                                 feature.tobytes()).hexdigest()})
    bank_array = np.asarray(bank)
    scene = next(row for row in json.loads(V192.read_text())["rows"]
                 if row["seed"] == 103)
    base_positions = scene["audits"]["0.065"]["positions_xy_m"]
    memory = np.asarray(trace["final_actor_target_xy_m"], dtype=float)
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
    estimated = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = []
    with imageio.get_writer(OUT / "nearby-absent-distractors.mp4", fps=4,
                            codec="libx264", quality=7) as video:
        for moved in ("CreamCheese", "Parmesan"):
            for offset in OFFSETS_M:
                case = f"{moved}-dy{int(100 * offset):02d}cm"
                positions = {"CreamCheese": tuple(base_positions["CreamCheese"]),
                             "Parmesan": tuple(base_positions["Parmesan"])}
                positions[moved] = tuple(memory + np.array((0., offset)))
                room = MovedPropRoom(positions)
                try:
                    room.reset()
                    room.layout(103, absent=True, furnished=True)
                    room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
                    room.set_table_height(.37)
                    room.data.qpos[:] = final_qpos
                    room.actual[:] = actual
                    room.estimated[:] = estimated
                    room._forward()
                    gid = room.model.geom(f"loop_{moved}_geom").id
                    other_ids = [other for other in range(room.model.ngeom)
                                 if room.model.geom(other).name.startswith("loop_")
                                 and room.model.geom(other).name != "loop_rug_visual"
                                 and other != gid]
                    scene_gap, closest = min((float(mujoco.mj_geomDistance(
                        room.model, room.data, gid, other, 10., None)),
                        room.model.geom(other).name) for other in other_ids)
                    valid = scene_gap >= .03
                    cases.append({"case": case, "moved": moved,
                                  "offset_y_m": offset, "moved_xy_m": positions[moved],
                                  "minimum_other_scene_gap_m": round(scene_gap, 5),
                                  "closest_other_scene_geom": closest,
                                  "valid_scene": valid})
                    if not valid:
                        print(case, "invalid geometry", scene_gap, flush=True)
                        continue
                    for yaw in ANGLES_RAD:
                        row, image = score_view(room, vision, bank_array, memory,
                                                yaw, case)
                        rows.append(row)
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, image.width, 26), fill=(0, 0, 0))
                        draw.text((6, 6), f"ABSENT {case} yaw {yaw:.1f} | bank accepts {row['view_bank_accepted_box'] is not None}", fill=(255, 255, 255))
                        video.append_data(np.asarray(image))
                    print(case, "valid", "bank accepts", sum(
                        row["view_bank_accepted_box"] is not None for row in rows
                        if row["case"] == case), flush=True)
                finally:
                    room.close()
    valid_cases = sum(case["valid_scene"] for case in cases)
    bank_false = sum(row["view_bank_accepted_box"] is not None for row in rows)
    owner_false = sum(row["fixed_owner_accepted_box"] is not None for row in rows)
    result = {"scope": "V199 revealed seed103 target-absent near-memory existing CreamCheese/Parmesan placements, offsets0/4/8cm, yaw0-1.4/0.2, final V198 online owner-anchored two-crop bank; frozen0.35/12cm gates. Geometry-valid >=3cm cases only; evaluator object identity after actor decision. Stationary optical negative control, not broad lookalike validation.",
              "source_sha256": sha(SOURCE), "v198_sha256": sha(V198),
              "v192_sha256": sha(V192), "owner_sha256": sha(OWNER),
              "bank_sources": bank_sources,
              "cases": cases,
              "valid_cases": valid_cases,
              "valid_views": len(rows),
              "fixed_owner_false_acceptances": owner_false,
              "view_bank_false_acceptances": bank_false,
              "negative_control_pass": valid_cases > 0 and bank_false == 0,
              "video_sha256": sha(OUT / "nearby-absent-distractors.mp4"),
              "rows": rows}
    (OUT / "nearby-absent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("rows", "cases", "bank_sources")}, indent=2))


if __name__ == "__main__":
    main()
