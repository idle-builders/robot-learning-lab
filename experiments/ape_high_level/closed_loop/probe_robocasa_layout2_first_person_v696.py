"""Replay the initial owner view with a released RGB-D person detector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_owner_reacquire_rgbd_v560 as person_detector
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_person_joined_v497 as joined
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-layout2-first-person-v696"
BASELINE = transfer.OUT / "report.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episode = json.loads(BASELINE.read_text())["episode"]
    first = episode["route_episode"]["trace"][0]
    model = person.scene_with_person(transfer.PROP_START, layout_id=2)
    data = mujoco.MjData(model)
    shifted_torso = interaction.TORSO_XY + transfer.OWNER_SHIFT
    shifted_rest = interaction.REST_HAND + [*transfer.OWNER_SHIFT, 0.]
    person.set_owner_pose(model, shifted_torso, shifted_rest, show_can=False)
    actual = np.asarray(first["actual_xy_m"])
    estimate = np.asarray(first["estimated_xy_m"])
    model.body_pos[model.body("audit_ape").id, :2] = actual
    model.cam_pos[model.camera("head").id, :2] = actual
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    rgb, depth, capture_wall = owner.capture(renderer, data, "head", True)
    renderer.close()
    image_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
    assert image_hash == first["rgb_sha256"]
    Image.fromarray(rgb).save(OUT / "initial-head-rgb.png")
    np.savez_compressed(OUT / "initial-head-depth.npz", depth=depth)
    observation = person_detector.observe(rgb, depth, data, model, "head",
                                          estimate, actual)
    proposed = None
    if observation["status"] == "person_surface_localized":
        surface = np.asarray(observation["point_xyz_est_m"])
        proposed = (surface[:2] + [0., -.8]).round(5).tolist()
    report = {"schema": "robot-learning-lab.layout2-first-person-replay.v1",
              "baseline_report_sha256": joined.route.sha256(BASELINE),
              "source_sha256": joined.route.sha256(SOURCE),
              "detector_weights_sha256": joined.route.sha256(
                  person_detector.DETECTOR / "model.safetensors"),
              "capture_time_s": first["time_s"],
              "capture_wall_s": capture_wall,
              "rgb_sha256": image_hash,
              "depth_file": str(OUT / "initial-head-depth.npz"),
              "actor_estimated_xy_m": estimate.tolist(),
              "observation": observation,
              "proposed_owner_view_xy_m": proposed,
              "candidate_rule": "0.8 m south of person surface point in estimated world frame; route map admission and second-view verification not yet run",
              "limits": "Single initial frame; prompt box is not owner identity, and camera/depth timing is replayed rather than run concurrently with guard."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(observation["status"], observation.get("detections"),
          "point", observation.get("point_xyz_est_m"),
          "candidate", proposed, flush=True)


if __name__ == "__main__":
    main()
