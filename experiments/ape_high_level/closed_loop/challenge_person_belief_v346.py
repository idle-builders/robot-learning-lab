"""V346: frozen person belief under shifted people, lighting and hard negatives.

The actor receives rendered upper/lower RGB, Grounding DINO boxes and capture
times. Person geometry and scenario labels score the belief after its decision.
The prior V345 thresholds and temporal rules are unchanged.
"""

from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (AutoModelForZeroShotObjectDetection,
                          AutoProcessor)

import direct_structural_rooms_v270 as structural
from active_scan import DETECTOR
from articulated_person_scene import (NEAR_HAND_XYZ_M, PERSON_GEOMS,
                                      START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_person_grounding_v344 import detect, evaluator_boxes
from direct_structural_rooms_v270 import selected_visual
from person_belief import PersonBelief
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v346"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
TIMES_S = (16.2, 16.4, 16.6)
POSITIVE_VARIANTS = (
    {"name": "base", "offset_xyz_m": (0., 0., 0.),
     "shirt_rgba": ".13 .38 .65 1", "light_scale": 1., "head_yaw_rad": 1.2},
    {"name": "left_dim", "offset_xyz_m": (0., .12, 0.),
     "shirt_rgba": ".46 .45 .43 1", "light_scale": .55, "head_yaw_rad": 1.0},
    {"name": "right_bright", "offset_xyz_m": (0., -.12, 0.),
     "shirt_rgba": ".19 .56 .28 1", "light_scale": 1.4, "head_yaw_rad": 1.2},
    {"name": "far_dark", "offset_xyz_m": (.12, 0., 0.),
     "shirt_rgba": ".13 .13 .17 1", "light_scale": .45, "head_yaw_rad": 1.35},
)
NEGATIVE_VARIANTS = (
    {"name": "plain_dim", "light_scale": .55, "head_yaw_rad": 1.0,
     "coatstand": False},
    {"name": "plain_bright", "light_scale": 1.4, "head_yaw_rad": 1.35,
     "coatstand": False},
    {"name": "coatstand", "light_scale": 1., "head_yaw_rad": 1.2,
     "coatstand": True},
)


def add_coatstand(base: mujoco.MjModel) -> mujoco.MjModel:
    """A static human-shaped jacket display, intentionally not a person."""
    visual = selected_visual()
    with tempfile.NamedTemporaryFile(suffix=".xml", dir=visual.parent,
                                     delete=False) as stream:
        saved = Path(stream.name)
    try:
        mujoco.mj_saveLastXML(str(saved), base)
        root = ET.parse(saved).getroot()
        body = ET.SubElement(root.find("worldbody"), "body",
                             name="loop_coatstand", pos="0 0 0")
        shared = {"group": "0", "contype": "0", "conaffinity": "0"}
        ET.SubElement(body, "geom", name="loop_coatstand_jacket_geom",
                      type="capsule", fromto=".68 .76 .24 .62 .70 .68",
                      size=".13", rgba=".13 .38 .65 1", **shared)
        ET.SubElement(body, "geom", name="loop_coatstand_hat_geom",
                      type="sphere", pos=".61 .69 .82", size=".11",
                      rgba=".68 .49 .38 1", **shared)
        saved.write_text(ET.tostring(root, encoding="unicode"))
        return mujoco.MjModel.from_xml_path(str(saved))
    finally:
        saved.unlink(missing_ok=True)


def review(lower: Image.Image, upper: Image.Image,
           belief: dict, label: str) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 510), (238, 239, 238))
    canvas.paste(lower, (0, 30))
    canvas.paste(upper, (640, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), f"{label} | belief {belief['person_presence']} | "
              f"limb {belief['lower_limb_state']}", fill=(20, 20, 20))
    return np.asarray(canvas)


def run_case(seed: int, variant: dict, positive: bool,
             processor, model, device: str) -> dict:
    trace_path = (V323 / f"seed{seed}-present" /
                  f"seed{seed}-present/episode-trace.json")
    trace = json.loads(trace_path.read_text())
    label = f"seed{seed}-{'person' if positive else 'negative'}-{variant['name']}"
    branch = OUT / label
    branch.mkdir(parents=True, exist_ok=True)
    original_model = structural.structural_model
    if positive:
        structural.structural_model = (
            lambda kind, side: add_person(
                original_model(kind, side),
                shirt_rgba=variant["shirt_rgba"]))
    elif variant["coatstand"]:
        structural.structural_model = (
            lambda kind, side: add_coatstand(original_model(kind, side)))
    try:
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command", branch=branch)
    finally:
        structural.structural_model = original_model
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimated_pose(trace)
        room.head = variant["head_yaw_rad"]
        room.data.qpos[room.model.joint(
            "left_shoulder_pitch").qposadr[0]] = -.8
        room.model.light_diffuse[:] *= variant["light_scale"]
        offset = np.asarray(variant["offset_xyz_m"]) if positive else None
        if positive:
            room.model.body_pos[room.model.body("loop_person").id] = offset
        robot_geoms = RobotGeometry(room.model).geom_ids
        minimum_person_gap = float("inf")
        samples = []
        frames = []
        belief = PersonBelief()
        for step in range(21):
            fraction = step / 20
            room.time_s = 16.2 + .4 * fraction
            room._update_moving_obstacle()
            if positive:
                hand = (START_HAND_XYZ_M + fraction *
                        (NEAR_HAND_XYZ_M - START_HAND_XYZ_M))
                set_person_pose(room.model, hand)
                room.model.body_pos[room.model.body(
                    "loop_TomatoSauce").id] += offset
            room._forward()
            if positive:
                minimum_person_gap = min(
                    minimum_person_gap,
                    gap_to_person(room, robot_geoms)[0])
            if step % 10:
                continue
            rgb = {}
            for camera in ("lower_rgb", "upper_rgb"):
                rgb[camera], _ = room.capture(camera, depth=False)
            upper = detect(processor, model, device, rgb["upper_rgb"],
                           "person .")
            lower = detect(processor, model, device, rgb["lower_rgb"],
                           "hand . arm .")
            state = belief.ingest(room.time_s, upper, lower)
            actor_packet = branch / f"actor-{step // 10:02d}.npz"
            np.savez_compressed(
                actor_packet,
                upper_rgb=np.asarray(rgb["upper_rgb"]),
                lower_rgb=np.asarray(rgb["lower_rgb"]),
                estimated_pose_xyyaw=room.estimated.copy(),
                actor_qpos=room.data.qpos.copy(),
                capture_time_s=room.time_s)
            truth_box = evaluator_boxes(room, "upper_rgb")["person"] if (
                positive) else None
            sample = {"time_s": round(room.time_s, 3),
                      "actor_packet_path": str(actor_packet.relative_to(ROOT)),
                      "actor_packet_sha256": sha(actor_packet),
                      "actor_upper_person_detections": upper,
                      "actor_lower_limb_detections": lower,
                      "actor_belief": state,
                      "evaluator_person_upper_box": truth_box}
            samples.append(sample)
            frames.append(review(rgb["lower_rgb"], rgb["upper_rgb"],
                                 state, f"V346 {label} t{room.time_s:.1f}"))
        video = branch / "person-belief-challenge.mp4"
        with imageio.get_writer(video, fps=2, codec="libx264",
                                quality=7) as writer:
            for frame in frames:
                writer.append_data(frame)
        row = {"case": label, "seed": seed,
               "positive_person": positive,
               "variant": variant,
               "evaluator_minimum_person_gap_m": round(minimum_person_gap, 5)
               if positive else None,
               "actor_samples": samples,
               "video_path": str(video.relative_to(ROOT)),
               "video_sha256": sha(video),
               "actor_final_person_state": samples[-1]["actor_belief"][
                   "person_presence"]}
        print(label, row["actor_final_person_state"],
              row["evaluator_minimum_person_gap_m"], flush=True)
        return row
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR).to(device).eval()
    rows = []
    for seed in SEEDS:
        for variant in POSITIVE_VARIANTS:
            rows.append(run_case(seed, variant, True,
                                 processor, model, device))
        for variant in NEGATIVE_VARIANTS:
            rows.append(run_case(seed, variant, False,
                                 processor, model, device))
    report = {
        "scope": "V346 frozen V345 actor person belief under four articulated-person variations per seed (shift, shirt, head yaw and simulated light) and three no-person negatives per seed including a static jacket/hat stand. The Grounding DINO weights, prompts and PersonBelief thresholds are unchanged. Actor sees only rendered RGB, detector boxes, estimated pose/joints and timestamps; evaluator semantic labels and person clearances score outcomes. The geometry/light matrix is still procedural and kinematic, not a real human distribution or hardware camera validation.",
        "source_sha256": sha(SOURCE),
        "person_scene_source_sha256": sha(Path(add_person.__code__.co_filename)),
        "belief_source_sha256": sha(Path(__file__).with_name(
            "person_belief.py")),
        "detector_weights_sha256": sha(DETECTOR / "model.safetensors"),
        "selected_visual_sha256": sha(selected_visual()),
        "rows": rows,
        "score": {
            "positive_confirmed_at_end": sum(
                row["positive_person"] and row["actor_final_person_state"] ==
                "confirmed" for row in rows),
            "negative_confirmed_at_end": sum(
                not row["positive_person"] and row[
                    "actor_final_person_state"] == "confirmed" for row in rows),
            "positive_cases": sum(row["positive_person"] for row in rows),
            "negative_cases": sum(not row["positive_person"] for row in rows)}}
    (OUT / "person-belief-challenge.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
