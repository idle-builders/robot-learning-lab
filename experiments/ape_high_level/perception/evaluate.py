"""Run fresh ape camera episodes and retain images, memory, and truth-only scores."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from detect import detect, frame_valid
from memory import SceneMemory
from scene import ApeScene, NAMES, ROOT


PHASES = (
    ("initial", 0.0, 0.0, False, False, False),
    ("look_left", 0.9, 0.0, False, False, False),
    ("look_right", -0.9, 0.0, False, False, False),
    ("look_down", 0.0, 0.2, False, False, False),
    ("reacquire", 0.0, 0.0, False, False, False),
    ("occluded", 0.0, 0.0, True, False, False),
    ("blackout_start", 0.0, 0.0, False, False, True),
    ("blackout_mid", 0.0, 0.0, False, False, True),
    ("blackout_stale", 0.0, 0.0, False, False, True),
    ("camera_returns", 0.0, 0.0, False, False, False),
    ("duplicate_red", 0.0, 0.0, False, True, False),
    ("duplicate_removed", 0.0, 0.0, False, False, False),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotate(rgb, detections, memory, path: Path):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for d in detections:
        x, y, w, h = d.bbox_xywh
        draw.rectangle((x, y, x + w, y + h), outline="white", width=2)
        draw.text((x, max(0, y - 12)), d.name, fill="white", stroke_width=1,
                  stroke_fill="black")
    labels = "; ".join(f"{name}: {track.status}" for name, track in memory.items())
    draw.rectangle((0, 0, image.width, 15), fill="black")
    draw.text((5, 2), labels, fill="white")
    image.save(path)


def _evaluator_visibility(scene: ApeScene) -> dict[str, dict]:
    """Rendered geom IDs are read only after the RGB perception update."""
    scene.renderer.enable_segmentation_rendering()
    try:
        scene.renderer.update_scene(scene.data, camera="ape_eye")
        segmentation = scene.renderer.render().copy()
    finally:
        scene.renderer.disable_segmentation_rendering()
    visible = {}
    for name in (*NAMES, "duplicate red cube"):
        body_name = "duplicate_red" if name == "duplicate red cube" else name.replace(" ", "_")
        body_id = scene.model.body(body_name).id
        geom_id = scene.model.body_geomadr[body_id]
        mask = (
            (segmentation[:, :, 0] == geom_id)
            & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
        )
        visible[name] = {
            "pixels": int(np.count_nonzero(mask)),
            "touches_image_edge": bool(
                mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any()
            ),
        }
    return visible


def run_episode(seed: int, requested_target: str, output: Path) -> dict:
    scene = ApeScene(seed)
    memory = SceneMemory()
    frames = []
    errors = []
    uncertainty_covered = []
    false_identities = 0
    visible_objects = Counter()
    detected_visible_objects = Counter()
    fully_visible_objects = Counter()
    detected_fully_visible_objects = Counter()
    try:
        for step, (phase, yaw, pitch, blocked, duplicate, blackout) in enumerate(PHASES):
            time_s = step * 0.3
            scene.set_head(yaw, pitch)
            scene.set_blocker(blocked)
            scene.set_duplicate_red(duplicate)
            pose = scene.camera_pose()
            rgb = np.zeros((360, 640, 3), dtype=np.uint8) if blackout else scene.image()
            detections = detect(rgb, pose, time_s)
            camera_ok = frame_valid(rgb)
            tracks = memory.update(detections, time_s, pose, frame_valid=camera_ok)
            # The evaluator alone reads prop poses. The detector and memory
            # above receive only RGB, camera pose, and time.
            truth = scene.truth()
            visible_pixels = _evaluator_visibility(scene)
            if duplicate:
                truth["duplicate red cube"] = scene.world_to_base_xy(
                    scene.model.body_pos[scene.model.body("duplicate_red").id, :2]
                )
            evaluations = []
            for d in detections:
                matches = [truth[d.name]]
                if d.name == "red cube" and duplicate:
                    matches.append(truth["duplicate red cube"])
                error = min(float(np.linalg.norm(np.subtract(d.xy_m, xy))) for xy in matches)
                nearest_other = min(
                    float(np.linalg.norm(np.subtract(d.xy_m, xy)))
                    for other, xy in truth.items()
                    if other != d.name and not (d.name == "red cube" and other == "duplicate red cube")
                )
                wrong = nearest_other + 0.01 < error or error > 0.05
                false_identities += int(wrong)
                if not wrong:
                    errors.append(error)
                    uncertainty_covered.append(error <= d.uncertainty_m)
                evaluations.append({"name": d.name, "error_m": error,
                                    "uncertainty_m": d.uncertainty_m,
                                    "identity_error": wrong})
            if not blackout:
                for instance in (*NAMES, "duplicate red cube"):
                    minimum = 50 if instance != "yellow destination" else 80
                    visibility = visible_pixels[instance]
                    if visibility["pixels"] < minimum:
                        continue
                    label = "red cube" if instance == "duplicate red cube" else instance
                    visible_objects[label] += 1
                    if not visibility["touches_image_edge"]:
                        fully_visible_objects[label] += 1
                    matched = any(d.name == label and np.linalg.norm(
                        np.subtract(d.xy_m, truth[instance])
                    ) <= 0.05 for d in detections)
                    if matched:
                        detected_visible_objects[label] += 1
                        if not visibility["touches_image_edge"]:
                            detected_fully_visible_objects[label] += 1
            filename = f"seed-{seed:04d}-{step:02d}-{phase}.png"
            _annotate(rgb, detections, tracks, output / filename)
            frames.append({
                "phase": phase, "time_s": time_s, "head_yaw_rad": yaw,
                "head_pitch_rad": pitch, "camera_blackout": blackout,
                "camera_frame_valid": camera_ok,
                "blocker_enabled": blocked, "duplicate_red_enabled": duplicate,
                "requested_target": requested_target,
                "requested_target_lookup": memory.locate(requested_target).record(),
                "camera_position_m": pose[0].tolist(),
                "camera_rotation": pose[1].tolist(),
                "base_world_xy_m": scene.data.qpos[:2].tolist(),
                "base_world_quat_wxyz": scene.data.qpos[3:7].tolist(),
                "image": filename,
                "detections": [d.record() for d in detections],
                "memory": memory.record(),
                "evaluation_only": {"truth_xy_m": {k: v.tolist() for k, v in truth.items()},
                                    "visible_geoms": visible_pixels,
                                    "detections": evaluations},
            })
    finally:
        scene.close()

    initial = frames[0]
    by_phase = {frame["phase"]: frame for frame in frames}
    visible_initial = Counter(d["name"] for d in initial["detections"])
    red = lambda phase: by_phase[phase]["memory"]["red cube"]
    gates = {
        "initial_four_unique": all(visible_initial[name] == 1 for name in NAMES),
        "requested_target_located": initial["requested_target_lookup"]["usable"],
        "look_away_hides_an_object": any(
            any(t["status"] == "out_of_view" for t in by_phase[p]["memory"].values())
            for p in ("look_left", "look_right")
        ),
        "red_occluded": red("occluded")["status"] == "occluded_or_missed",
        "stale_blackout_unusable": all(
            not t["usable"] and t["status"] in ("stale", "never_seen")
            for t in by_phase["blackout_stale"]["memory"].values()
        ),
        "blackout_never_usable": all(
            not t["usable"] for frame in frames if frame["camera_blackout"]
            for t in frame["memory"].values()
        ),
        "red_reacquired": red("camera_returns")["status"] == "visible",
        "duplicate_red_ambiguous": red("duplicate_red")["status"] == "ambiguous"
                                   and not red("duplicate_red")["usable"],
        "red_unique_again": red("duplicate_removed")["status"] == "visible",
        "first_seen_preserved": all(
            by_phase["duplicate_removed"]["memory"][name]["first_seen_s"] == 0.0
            for name in NAMES
        ),
        "visible_detection_recall_at_least_90pct": (
            sum(detected_fully_visible_objects.values())
            >= 0.9 * sum(fully_visible_objects.values())
        ),
    }
    reacquisition_s = []
    look_away_reacquisition_s = []
    for name in NAMES:
        lost_at = None
        lost_out_of_view = False
        for frame in frames:
            status = frame["memory"][name]["status"]
            if status in ("out_of_view", "occluded_or_missed") and lost_at is None:
                lost_at = frame["time_s"]
                lost_out_of_view = status == "out_of_view"
            elif status == "visible" and lost_at is not None:
                reacquisition_s.append(frame["time_s"] - lost_at)
                if lost_out_of_view:
                    look_away_reacquisition_s.append(frame["time_s"] - lost_at)
                lost_at = None
                lost_out_of_view = False
    usable_without_detection = sum(
        track["usable"] and not any(d["name"] == name for d in frame["detections"])
        for frame in frames if frame["camera_blackout"] or frame["blocker_enabled"]
        for name, track in frame["memory"].items()
    )
    return {
        "seed": seed, "requested_target": requested_target,
        "body_sha256": scene.source_sha256,
        "frames": frames, "gates": gates,
        "visible_position_errors_m": errors,
        "within_declared_uncertainty": sum(uncertainty_covered),
        "evaluated_positions": len(uncertainty_covered),
        "identity_errors": false_identities,
        "visible_objects": dict(visible_objects),
        "detected_visible_objects": dict(detected_visible_objects),
        "fully_visible_objects": dict(fully_visible_objects),
        "detected_fully_visible_objects": dict(detected_fully_visible_objects),
        "reacquisition_delays_s": reacquisition_s,
        "look_away_reacquisition_delays_s": look_away_reacquisition_s,
        "usable_without_current_detection": usable_without_detection,
        "pass": all(gates.values()) and false_identities == 0
                and all(uncertainty_covered)
                and (not errors or statistics.median(errors) <= 0.05),
    }


def publish_dashboard(output: Path, episodes: list[dict], summary: dict) -> None:
    relative = output.resolve().relative_to(ROOT)
    assets = [{"id": "trace", "title": "Full camera trace and truth-only evaluation",
               "kind": "data", "path": str(relative / "trace.json")}]
    for episode in episodes:
        for frame in episode["frames"]:
            assets.append({"id": f"seed-{episode['seed']}-{frame['phase']}",
                           "title": f"Seed {episode['seed']} · {frame['phase']}",
                           "kind": "image", "path": str(relative / frame["image"])})
    matrix = [{"seed": ep["seed"], "requested_target": ep["requested_target"],
               "identity_errors": ep["identity_errors"],
               "median_visible_error_m": statistics.median(ep["visible_position_errors_m"])
               if ep["visible_position_errors_m"] else None,
               "reacquisition_delays_s": ep["reacquisition_delays_s"]}
              for ep in episodes]
    record = {
        "id": output.name, "title": "Camera memory · " + output.name,
        "workspace": "Perception", "area": "Camera memory", "status": "historical_35cm",
        "observation": "Stationary-base camera episodes tracked colored cubes through head turns, occlusion and blackout; see each frame and full trace for where identity or availability degraded.",
        "measurements": f"{summary['episodes']} episodes; median visible position error {summary['median_visible_error_m']:.4f} m; {summary['identity_errors']} identity errors; reacquisition delays and each episode in the case table.",
        "limitations": "Historical 35 cm stationary-base scene with colored props. Truth-only scoring does not establish current 50 cm perception or physical performance.",
        "provenance": [str(relative / "trace.json")], "matrix": matrix,
        "sections": [{"title": "Per-episode conditions and checks", "payload":
                      [{"seed": ep["seed"], "gates": ep["gates"], "scope": summary["scope"]}
                       for ep in episodes]}],
        "assets": assets,
    }
    subprocess.run([sys.executable, str(ROOT / "tools" / "results_hub.py"), "publish", "-"],
                   input=json.dumps(record), text=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(142, 152)))
    parser.add_argument("--target", choices=NAMES[:3],
                        help="Request this cube in every episode; default rotates across all three")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts" / (
        "ape-perception-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:6]
    )
    output.mkdir(parents=True, exist_ok=False)
    episodes = [run_episode(seed, args.target or NAMES[i % 3], output)
                for i, seed in enumerate(args.seeds)]
    errors = [error for ep in episodes for error in ep["visible_position_errors_m"]]
    summary = {
        "episodes": len(episodes),
        "passed_episodes": sum(ep["pass"] for ep in episodes),
        "median_visible_error_m": statistics.median(errors) if errors else None,
        "max_visible_error_m": max(errors) if errors else None,
        "identity_errors": sum(ep["identity_errors"] for ep in episodes),
        "within_declared_uncertainty": sum(ep["within_declared_uncertainty"] for ep in episodes),
        "evaluated_positions": sum(ep["evaluated_positions"] for ep in episodes),
        "visible_objects": dict(sum((Counter(ep["visible_objects"]) for ep in episodes), Counter())),
        "detected_visible_objects": dict(sum(
            (Counter(ep["detected_visible_objects"]) for ep in episodes), Counter()
        )),
        "fully_visible_objects": dict(sum(
            (Counter(ep["fully_visible_objects"]) for ep in episodes), Counter()
        )),
        "detected_fully_visible_objects": dict(sum(
            (Counter(ep["detected_fully_visible_objects"]) for ep in episodes), Counter()
        )),
        "reacquisition_delays_s": [delay for ep in episodes
                                   for delay in ep["reacquisition_delays_s"]],
        "look_away_reacquisition_delays_s": [delay for ep in episodes
                                             for delay in ep["look_away_reacquisition_delays_s"]],
        "usable_without_current_detection": sum(
            ep["usable_without_current_detection"] for ep in episodes
        ),
        "source_sha256": episodes[0]["body_sha256"] if episodes else None,
        "code_sha256": {p.name: _sha(p) for p in Path(__file__).parent.glob("*.py")},
        "camera": {"width": 640, "height": 360, "fovy_deg": 67.0,
                   "pose_source": "MuJoCo camera FK from fixed base and head joint state"},
        "scope": "stationary-base RGB, three named colored cubes, yellow destination",
    }
    (output / "trace.json").write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2) + "\n")
    publish_dashboard(output, episodes, summary)
    print(json.dumps({"output": str(output), "summary": summary}))


if __name__ == "__main__":
    main()
