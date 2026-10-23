"""Screen provisional Direct Frame camera positions in captured owner rooms.

The actor sees only upper RGB-D, camera-projected near-range and a supplied
person click. Evaluator-only background subtraction scores initial visibility.
This is an optical/sensing screen, not a body-qualified navigation episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SCAN = ROOT / "experiments/ape_high_level/real_scan_benchmark"
PEOPLE = ROOT / "experiments/ape_high_level/realistic_people"
sys.path.insert(0, str(SCAN))
sys.path.insert(0, str(PEOPLE))
from real_capture_dynamic_scene import ScannedSittingPerson, TexturedCaptureScene  # noqa: E402
from rgbd_range_projection import projected_depth_fan  # noqa: E402
from run_real_capture_owner_loop import ANGLES, MODEL_ROOT, SCENES, owner_pose  # noqa: E402
from online_owner_actor import OnlineOwnerActor  # noqa: E402
from mounted_camera_geometry import optical_pixel_to_body, observed_free_body_fan  # noqa: E402


DEFAULT_OUT = ROOT / "artifacts/ape-sensors-50cm/captured-room-mounted-dual-v1"
DEFAULT_PLAN = DEFAULT_OUT / "plan.json"
CONTRACT = ROOT / "experiments/compact_morphology/ape_frame50/frame_contract.yaml"
PERSON = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"
DESIGN = ROOT / "robot/ape-current-design.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_pose(start_xz: tuple[float, float], floor_y: float,
                heading_deg: float, frame: dict, pitch_deg: float
                ) -> tuple[np.ndarray, np.ndarray]:
    heading = math.radians(heading_deg)
    forward = np.array([math.sin(heading), 0., math.cos(heading)])
    left = np.array([-math.cos(heading), 0., math.sin(heading)])
    up = np.array([0., 1., 0.])
    x, y, z = frame["position_m"]
    eye = np.array([start_xz[0], floor_y, start_xz[1]]) + \
          x * forward + y * left + z * up
    pitch = math.radians(pitch_deg)
    optical_forward = math.cos(pitch) * forward + math.sin(pitch) * up
    return eye, optical_forward


def visible_person_mask(with_person: np.ndarray,
                        background: np.ndarray) -> np.ndarray:
    return np.isfinite(with_person) & (
        ~np.isfinite(background) | (with_person + .025 < background))


def save_depth(path: Path, depth: np.ndarray) -> None:
    millimetres = np.where(np.isfinite(depth),
                           np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)
    Image.fromarray(millimetres).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    output = args.output.resolve()
    plan = json.loads(plan_path.read_text())
    contract = yaml.safe_load(CONTRACT.read_text())["sensors"]["cameras"]
    upper = contract["upper_frame"]
    lower = contract["lower_frame"]
    vfov = float(contract["field_of_view_deg"]["vertical"])
    focal = 360 / (2 * math.tan(math.radians(vfov) / 2))
    hfov = 2 * math.degrees(math.atan(640 / (2 * focal)))
    design = json.loads(DESIGN.read_text())
    person = ScannedSittingPerson(PERSON)
    source_files = [Path(__file__), plan_path, CONTRACT, DESIGN, PERSON,
                    ROOT / "experiments/ape_high_level/sensors/cove_sensor_package.py",
                    SCAN / "real_capture_dynamic_scene.py",
                    SCAN / "rgbd_range_projection.py",
                    ROOT / "experiments/ape_high_level/sensors/mounted_camera_geometry.py",
                    PEOPLE / "online_owner_actor.py"]
    rows = []
    for scene_id in plan["scenes"]:
        config = SCENES[scene_id]
        source_trace = config["trace"]
        source = json.loads(source_trace.read_text())
        source_mesh = ROOT / source["source_mesh"]
        scene = TexturedCaptureScene(source_mesh,
                                     floor_y=source["floor_y_1pct_m"],
                                     scale=source.get("scale_to_metres", 1.),
                                     scanned_person=person,
                                     person_facing_deg=config["person_facing_deg"])
        owner_xz = owner_pose(0., config)
        upper_eye, upper_forward = camera_pose(
            config["start_xz"], scene.floor_y, config["heading_deg"],
            upper, float(upper["pitch_up_deg"]))
        lower_eye, lower_forward = camera_pose(
            config["start_xz"], scene.floor_y, config["heading_deg"],
            lower, -float(lower["pitch_down_deg"]))
        upper_rgb, upper_depth = scene.observe(
            upper_eye, upper_forward, owner_xz=owner_xz, vfov_deg=vfov)
        _, background_depth = scene.observe(
            upper_eye, upper_forward, owner_xz=None, vfov_deg=vfov)
        lower_rgb, lower_depth = scene.observe(
            lower_eye, lower_forward, owner_xz=owner_xz, vfov_deg=vfov)
        mask = visible_person_mask(upper_depth, background_depth)
        ys, xs = np.where(mask)
        click = None
        if len(xs):
            center = np.array([np.median(xs), np.median(ys)])
            index = int(np.argmin((xs - center[0]) ** 2 + (ys - center[1]) ** 2))
            click = (int(xs[index]), int(ys[index]))
        upper_fan = projected_depth_fan(
            upper_depth, ANGLES, vfov_deg=vfov,
            eye_height_m=float(upper["position_m"][2]),
            camera_pitch_down_deg=-float(upper["pitch_up_deg"]))
        lower_fan = projected_depth_fan(
            lower_depth, ANGLES, vfov_deg=vfov,
            eye_height_m=float(lower["position_m"][2]),
            camera_pitch_down_deg=float(lower["pitch_down_deg"]))
        camera_fan = np.fmin(upper_fan, lower_fan)
        body_fan = observed_free_body_fan((
            (upper_depth, tuple(upper["position_m"]), float(upper["pitch_up_deg"])),
            (lower_depth, tuple(lower["position_m"]),
             -float(lower["pitch_down_deg"]))), ANGLES, vfov_deg=vfov)
        actor = None
        if click is not None:
            vision = OnlineOwnerActor(MODEL_ROOT / "grounding-dino-tiny",
                                      MODEL_ROOT / "dinov2-small",
                                      click_xy=click, device="mps")
            actor = vision.observe(upper_rgb, upper_depth, camera_fan,
                                   ANGLES, time_s=0., fov_deg=hfov)
        folder = output / scene_id
        folder.mkdir(parents=True, exist_ok=True)
        marked = upper_rgb.copy()
        if click is not None:
            cv2.drawMarker(marked, click, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        Image.fromarray(marked).save(folder / "upper_owner.png")
        Image.fromarray(lower_rgb).save(folder / "lower_route.png")
        save_depth(folder / "upper_depth_mm.png", upper_depth)
        save_depth(folder / "lower_depth_mm.png", lower_depth)
        proposals = [] if actor is None else actor.get("proposals", [])
        matches = [i for i, proposal in enumerate(proposals)
                   if click is not None and
                   proposal["box_xyxy"][0] <= click[0] <= proposal["box_xyxy"][2] and
                   proposal["box_xyxy"][1] <= click[1] <= proposal["box_xyxy"][3]]
        body_measurement = None
        if actor is not None and actor.get("chosen_proposal") is not None and \
                actor.get("local_depth_m") is not None:
            box = proposals[actor["chosen_proposal"]]["box_xyxy"]
            u = (box[0] + box[2]) / 2
            v = (box[1] + box[3]) / 2
            point = optical_pixel_to_body(
                u, v, float(actor["local_depth_m"]),
                tuple(upper["position_m"]), pitch_up_deg=float(upper["pitch_up_deg"]),
                vfov_deg=vfov)
            body_measurement = {
                "visible_surface_body_xyz_m": list(point),
                "horizontal_range_from_body_m": float(math.hypot(point[0], point[1])),
                "bearing_left_deg": float(math.degrees(math.atan2(point[1], point[0]))),
                "source_box_center_px": [u, v],
                "source_optical_depth_m": float(actor["local_depth_m"]),
                "not_person_center": True}
        row = {"scene_id": scene_id, "source_trace": str(source_trace.relative_to(ROOT)),
               "source_trace_sha256": digest(source_trace),
               "source_mesh_sha256": digest(source_mesh),
               "owner_start_xz_m": list(config["owner_xz"]),
               "upper_eye_xyz_m": upper_eye.tolist(),
               "upper_forward_xyz": upper_forward.tolist(),
               "lower_eye_xyz_m": lower_eye.tolist(),
               "lower_forward_xyz": lower_forward.tolist(),
               "visible_owner_pixels_upper_evaluator_only": int(len(xs)),
               "owner_click_xy_evaluator_selected": list(click) if click is not None else None,
               "actor_proposals_covering_click": matches,
               "actor_decision": actor,
               "body_frame_visible_surface_diagnostic": body_measurement,
               "known_low_beams": int((~np.isnan(camera_fan[0])).sum()),
               "known_mid_beams": int((~np.isnan(camera_fan[1])).sum()),
               "known_body_frame_low_free_beams": int((~np.isnan(body_fan[0])).sum()),
               "known_body_frame_mid_free_beams": int((~np.isnan(body_fan[1])).sum()),
               "body_frame_free_mask_by_height": (~np.isnan(body_fan)).tolist(),
               "upper_rgb": str((folder / "upper_owner.png").relative_to(ROOT)),
               "lower_rgb": str((folder / "lower_route.png").relative_to(ROOT)),
               "upper_depth": str((folder / "upper_depth_mm.png").relative_to(ROOT)),
               "lower_depth": str((folder / "lower_depth_mm.png").relative_to(ROOT))}
        rows.append(row)
        print(scene_id, "owner px", len(xs), "click proposals", len(matches),
              "actor", actor["intent"] if actor else "none",
              "free beams", row["known_low_beams"], row["known_mid_beams"], flush=True)
        (output / "report.json").write_text(json.dumps({
            "schema": "ape.captured-room-mounted-dual-screen.v1",
            "plan": str(plan_path.relative_to(ROOT)), "plan_sha256": digest(plan_path),
            "current_design_status": design["status"],
            "camera_contract": {"upper": upper, "lower": lower,
                                "vfov_deg": vfov, "resolution_px": [640, 360]},
            "source_hashes": {str(p.relative_to(ROOT)): digest(p)
                              for p in source_files},
            "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
