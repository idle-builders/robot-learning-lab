"""Continue confirmed real-room finds into a guarded, reobserved approach.

The actor receives the confirmed candidate's saved RGB-D location, commanded
pose/yaw and fresh two-plane range. The simulator alone holds true geometry,
object placement and collision checks. This remains a kinematic proxy trial.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import (SCENES, TexturedCaptureScene,
    TexturedFloorObject, camera)
import run_real_capture_colour_bank_search as colour_bank


SOURCE = ART / "integrated_owner_20_colour_bank_none_report.json"
PRECOMMIT = ART / "post_confirmation_20_approach_precommit.json"
REPORT = ART / "post_confirmation_20_approach_report.json"
ASSETS = ROOT / "assets/real_indoor_scenes/post_confirmation_20_approach_20261015"
STANDOFF_M = .55
STEP_M = .04
TURN_DEG = 15.
MAX_ACTIONS = 24
BODY_RADIUS_M = .17
SENSOR_MARGIN_M = .05


def signed_angle(angle: float) -> float:
    return (angle + 180.) % 360. - 180.


def find_confirmed_source(folder: Path) -> tuple[dict, dict]:
    episode = json.loads((folder / "find_present/episode_trace.json").read_text())
    if episode["action"] != "owner_confirmed_visual_identity":
        raise ValueError("Source item was not owner-confirmed")
    for stage in episode["stage_reports"]:
        answer = stage["owner_answer"]
        if answer and answer["action"].startswith("owner_confirmed"):
            trace = json.loads((ROOT / stage["trace"]).read_text())
            verified = trace["verification"]
            location = verified["candidate_localization"]
            if location["status"] != "located" or \
                    location.get("actor_map_target_xz_m") is None:
                raise ValueError("Confirmed item lacks RGB-D location")
            return episode, {"stage": stage, "trace": trace,
                             "location": location}
    raise ValueError("No confirmed stage")


def make_scene(case: dict, screen: dict) -> TexturedCaptureScene:
    cfg = SCENES[case["scene_key"]]
    source = json.loads(cfg["trace"].read_text())
    scene = TexturedCaptureScene(ROOT / source["source_mesh"],
        floor_y=source["floor_y_1pct_m"],
        scale=source.get("scale_to_metres", 1.),
        source_z_up=source.get("source_z_up", False))
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    scene.set_props({case["target"]: (
        TexturedFloorObject(props / f"{case['target']}.obj", scale=.012),
        tuple(screen["find_target_xz"]), 0.),
        case["distractor"]: (
        TexturedFloorObject(props / f"{case['distractor']}.obj", scale=.012),
        tuple(screen["find_distractor_xz"]), 0.)})
    return scene


def choose_action(actor_world: np.ndarray, actor_yaw: float,
                  target_world: np.ndarray, ranges: np.ndarray
                  ) -> tuple[str, float, str]:
    delta = target_world - actor_world
    distance = float(np.linalg.norm(delta))
    if distance <= STANDOFF_M:
        return "hold", 0., "observed_target_standoff"
    bearing = math.degrees(math.atan2(delta[0], delta[1]))
    turn = signed_angle(bearing - actor_yaw)
    if abs(turn) > 5.:
        return "turn", float(np.clip(turn, -TURN_DEG, TURN_DEG)), "face_observed_target"
    low, mid = ranges
    forward = np.abs(np.arange(-90., 91., 5.)) <= 15.
    values = np.concatenate((low[forward], mid[forward]))
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return "hold", 0., "forward_range_unavailable"
    step = min(STEP_M, distance - STANDOFF_M)
    if float(finite.min()) < BODY_RADIUS_M + SENSOR_MARGIN_M + step:
        return "hold", 0., "forward_range_blocked"
    return "forward", float(step), "bounded_observed_target_approach"


def save_frame(out: Path, writer, frame: int, rgb: np.ndarray,
               depth: np.ndarray) -> None:
    Image.fromarray(rgb).save(out / f"rgb_{frame:03d}.png")
    Image.fromarray(np.where(np.isfinite(depth),
        np.clip(depth * 1000., 0., 65535.), 0).astype(np.uint16)).save(
            out / f"depth_{frame:03d}.png")
    writer.append_data(rgb)


def run_case(case: dict) -> dict:
    source_folder = (ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015"
                     / case["scene_key"])
    episode, confirmed = find_confirmed_source(source_folder)
    owner = json.loads((source_folder / "owner_approach/trace.json").read_text())
    last = owner["trace"][-1]
    screen_data = json.loads((ROOT / case["screen"]).read_text())
    screen = next(x for x in screen_data["scenes"] if x.get("scene_key") ==
                  case["scene_key"])
    scene = make_scene(case, screen)
    actual = np.asarray(last["evaluator_actual_xz_m"], dtype=float)
    actual_yaw = float(last["evaluator_actual_heading_deg"])
    start = np.asarray(SCENES[case["scene_key"]]["start_xz"], dtype=float)
    actor_world = start + np.asarray(last["actor_pose_xy_m"], dtype=float)
    actor_yaw = float(last["actor_heading_deg"])
    observed_target = np.asarray(confirmed["location"]["actor_map_target_xz_m"],
                                 dtype=float)
    out = ASSETS / case["scene_key"]
    out.mkdir(parents=True, exist_ok=True)
    actions = []
    result = "action_cap"
    with imageio.get_writer(str(out / "approach.mp4"), fps=5, codec="libx264",
                            quality=7, macro_block_size=8) as writer:
        for frame in range(MAX_ACTIONS + 1):
            rgb, depth, _ids, ranges = camera(scene, actual, actual_yaw, None)
            save_frame(out, writer, frame, rgb, depth)
            kind, amount, reason = choose_action(actor_world, actor_yaw,
                                                  observed_target, ranges)
            record = {"frame": frame, "rgb": relative(out / f"rgb_{frame:03d}.png"),
                "depth": relative(out / f"depth_{frame:03d}.png"),
                "actor_pose_world_xz_m": actor_world.tolist(),
                "actor_heading_deg": actor_yaw,
                "observed_target_world_xz_m": observed_target.tolist(),
                "range_angles_deg": np.arange(-90., 91., 5.).tolist(),
                "range_low_m": [float(v) if np.isfinite(v) else None for v in ranges[0]],
                "range_mid_m": [float(v) if np.isfinite(v) else None for v in ranges[1]],
                "decision": kind, "reason": reason, "amount": amount,
                "evaluator_actual_xz_m": actual.tolist(),
                "evaluator_actual_heading_deg": actual_yaw,
                "evaluator_actual_target_gap_m": float(np.linalg.norm(actual -
                    np.asarray(screen["find_target_xz"], dtype=float))),
                "evaluator_pose_error_m": float(np.linalg.norm(actual - actor_world))}
            if kind == "hold":
                result = reason
            elif kind == "turn":
                actor_yaw += amount
                actual_yaw += amount
            else:
                heading = math.radians(actual_yaw)
                delta = amount * np.array([math.sin(heading), math.cos(heading)])
                candidate = actual + delta
                eye = np.array([candidate[0], scene.floor_y + .42, candidate[1]])
                candidate_ranges = scene.range_fan(eye, actual_yaw, owner_xz=None,
                    angles_deg=np.arange(-90., 91., 5.))
                values = candidate_ranges[np.isfinite(candidate_ranges)]
                accepted = bool(scene.floor_supported(*candidate) and len(values) and
                                values.min() >= BODY_RADIUS_M)
                record["evaluator_motion_accepted"] = accepted
                if accepted:
                    actual = candidate
                    actor_world += amount * np.array([
                        math.sin(math.radians(actor_yaw)),
                        math.cos(math.radians(actor_yaw))])
                else:
                    result = "simulator_rejected_motion"
            actions.append(record)
            if kind == "hold" or result == "simulator_rejected_motion":
                break
    reconfirm = None
    if result == "observed_target_standoff":
        baseline = colour_bank.conditional.baseline
        vision = baseline.Vision("generic-proposal")
        vision._teaching_frame_seen = True
        teach = json.loads((source_folder / "teach/trace.json").read_text())
        teach_rgb = np.asarray(Image.open(source_folder / "teach/rgb_000.png").convert("RGB"))
        exemplar = vision.feature([Image.fromarray(teach_rgb).crop(
            tuple(teach["selected_box_xyxy"]))])[0]
        colour_exemplar = baseline.colour_histogram(teach_rgb,
                                                    teach["selected_box_xyxy"])
        reconfirm = baseline.run_search(scene, vision, actual, actual_yaw,
            actor_world - start, actor_yaw, exemplar,
            float(teach["acceptance_similarity"]), True, out / "reconfirm",
            start_time_s=float(last["decision_time_s"] + 2.),
            search_xz=tuple(screen["find_target_xz"]),
            search_distractor_xz=tuple(screen["find_distractor_xz"]),
            owner_start_xz=tuple(start), target_name=case["target"],
            distractor_name=case["distractor"], identity_mode="multi_cue",
            colour_exemplar=colour_exemplar,
            colour_acceptance=float(teach["colour_acceptance_similarity"]))
    trace = {"schema": "ape.post-confirmation-approach.v1",
        "scene_key": case["scene_key"], "source_episode": relative(source_folder /
            "find_present/episode_trace.json"),
        "source_confirmed_stage": confirmed["stage"]["trace"],
        "source_confirmed_rgb": confirmed["trace"]["verification"]["rgb_file"],
        "observed_target_world_xz_m": observed_target.tolist(),
        "policy": {"standoff_m": STANDOFF_M, "max_step_m": STEP_M,
            "max_turn_deg": TURN_DEG, "max_actions": MAX_ACTIONS,
            "body_radius_m": BODY_RADIUS_M, "sensor_margin_m": SENSOR_MARGIN_M},
        "action_result": result, "actions": actions,
        "reconfirmation_action": reconfirm["action"] if reconfirm else None,
        "reconfirmation_video": reconfirm["video"] if reconfirm else None,
        "video": relative(out / "approach.mp4"),
        "limits": "Ideal yaw/commanded kinematic proxy and ideal two-plane range; simulator geometry checks score commands. No native gait, contact/force or grasp. Owner reply remains scripted."}
    (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": case["scene_key"], "trace": relative(out / "trace.json"),
        "video": trace["video"], "action_result": result,
        "reconfirmation_action": trace["reconfirmation_action"],
        "accepted_forward_steps": sum(x["decision"] == "forward" and x.get(
            "evaluator_motion_accepted") for x in actions),
        "final_actual_target_gap_m": actions[-1]["evaluator_actual_target_gap_m"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    cases = json.loads((ART / "integrated_owner_20_conditional_precommit.json").read_text())[
        "cases"]
    frozen = {"schema": "ape.post-confirmation-20-approach-precommit.v1",
        "source_report": relative(SOURCE), "source_report_sha256": sha(SOURCE),
        "cases": [{"scene_key": c["scene_key"], "capture_group": c["capture_group"],
                   "screen": c["screen"], "screen_sha256": sha(ROOT / c["screen"])}
                  for c in cases],
        "policy": {"standoff_m": STANDOFF_M, "max_step_m": STEP_M,
            "max_turn_deg": TURN_DEG, "max_actions": MAX_ACTIONS,
            "body_radius_m": BODY_RADIUS_M, "sensor_margin_m": SENSOR_MARGIN_M,
            "final_check": "fresh conditional RGB-D owner-confirmed visual reconfirmation"},
        "score": ["bounded_actions", "observed_range_stop", "simulator_motion_rejection",
                  "final_target_gap", "fresh_visual_reconfirmation", "aligned_video_depth_trace"],
        "limits": "Development-exposed rooms; prior confirmed item and observed RGB-D location. Kinematic proxy, ideal yaw/range, static props and scripted owner."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen approach experiment changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    if sha(SOURCE) != frozen["source_report_sha256"]:
        raise ValueError("Source changed")
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cases:
        if args.scene and case["scene_key"] != args.scene:
            continue
        if sha(ROOT / case["screen"]) != next(x["screen_sha256"] for x in
                frozen["cases"] if x["scene_key"] == case["scene_key"]):
            raise ValueError("Placement changed")
        path = ASSETS / case["scene_key"] / "trace.json"
        if not path.exists():
            if args.max_new and launched >= args.max_new:
                break
            row = run_case(case)
            launched += 1
        else:
            x = json.loads(path.read_text())
            row = {"scene_key": case["scene_key"], "trace": relative(path),
                "video": x["video"], "action_result": x["action_result"],
                "reconfirmation_action": x["reconfirmation_action"],
                "accepted_forward_steps": sum(a["decision"] == "forward" and a.get(
                    "evaluator_motion_accepted") for a in x["actions"]),
                "final_actual_target_gap_m": x["actions"][-1][
                    "evaluator_actual_target_gap_m"]}
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.post-confirmation-20-approach-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows), "rows": rows},
            indent=2) + "\n")
        print(row["scene_key"], row["action_result"], row["accepted_forward_steps"],
              row["reconfirmation_action"], flush=True)


if __name__ == "__main__":
    main()
