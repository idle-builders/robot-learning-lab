"""Paired sensor-coverage arms for the 20-room confirmed-item approach.

The target estimate and turn/step policy are identical. Only the forward
clearance packet changes: separate ideal two-height range, projected upright
head depth, or projected upright+45-degree-down head depth. Unknown coverage
stops motion. A successful arm must reproduce the saved baseline endpoint
before reusing that endpoint's fresh owner reconfirmation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from rgbd_range_projection import projected_depth_fan, dual_view_fan
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES, camera
import run_post_confirmation_approach_v2 as v2


base = v2.base
COHORT = ART / "integrated_owner_20_conditional_precommit.json"
BASELINE_AUDIT = ART / "post_confirmation_20_approach_v2_audit.json"
PRECOMMIT = ART / "item_approach_sensor_20_precommit.json"
REPORT = ART / "item_approach_sensor_20_report.json"
OUT = ROOT / "assets/real_indoor_scenes/item_approach_sensor_20_20261015"
ARMS = ("separate_range", "upright_rgbd", "dual_rgbd")
ANGLES = np.arange(-90., 91., 5., dtype=np.float32)
FRONT = np.abs(ANGLES) <= 15.


def source(case: dict) -> tuple[dict, dict, dict, Path]:
    key = case["scene_key"]
    folder = (ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015"
              / key)
    _episode, confirmed = base.find_confirmed_source(folder)
    owner = json.loads((folder / "owner_approach/trace.json").read_text())
    screen_data = json.loads((ROOT / case["screen"]).read_text())
    screen = next(x for x in screen_data["scenes"] if x.get("scene_key") == key)
    return confirmed, owner["trace"][-1], screen, folder


def encode_range(packet: np.ndarray) -> list[list[float | str | None]]:
    return [[None if np.isnan(x) else "clear" if np.isinf(x) else float(x)
             for x in plane] for plane in packet]


def choose(actor: np.ndarray, yaw: float, target: np.ndarray,
           packet: np.ndarray) -> tuple[str, float, str, dict]:
    delta = target - actor
    distance = float(np.linalg.norm(delta))
    front = packet[:, FRONT]
    health = {"unknown_front_sectors": int(np.isnan(front).sum()),
              "known_front_sectors": int((~np.isnan(front)).sum()),
              "finite_front_min_m": float(np.min(front[np.isfinite(front)]))
                  if np.isfinite(front).any() else None}
    if distance <= base.STANDOFF_M + v2.TOLERANCE_M:
        return "hold", 0., "observed_target_standoff", health
    bearing = math.degrees(math.atan2(delta[0], delta[1]))
    turn = base.signed_angle(bearing - yaw)
    if abs(turn) > 5.:
        return "turn", float(np.clip(turn, -base.TURN_DEG, base.TURN_DEG)), \
            "face_observed_target", health
    if np.isnan(front).any():
        return "hold", 0., "forward_clearance_unknown", health
    step = min(base.STEP_M, distance - base.STANDOFF_M)
    if float(front.min()) < base.BODY_RADIUS_M + base.SENSOR_MARGIN_M + step:
        return "hold", 0., "forward_range_blocked", health
    return "forward", float(step), "bounded_observed_target_approach", health


def run_arm(case: dict, arm: str) -> dict:
    confirmed, last, screen, source_folder = source(case)
    scene = base.make_scene(case, screen)
    key = case["scene_key"]
    actual = np.asarray(last["evaluator_actual_xz_m"], dtype=float)
    actual_yaw = float(last["evaluator_actual_heading_deg"])
    start = np.asarray(SCENES[key]["start_xz"], dtype=float)
    actor = start + np.asarray(last["actor_pose_xy_m"], dtype=float)
    actor_yaw = float(last["actor_heading_deg"])
    target = np.asarray(confirmed["location"]["actor_map_target_xz_m"], dtype=float)
    target_true = np.asarray(screen["find_target_xz"], dtype=float)
    output = OUT / key / arm
    output.mkdir(parents=True, exist_ok=True)
    actions = []
    result = "action_cap"
    with imageio.get_writer(str(output / "first_person.mp4"), fps=5,
            codec="libx264", quality=7, macro_block_size=8) as writer:
        for frame in range(base.MAX_ACTIONS + 1):
            rgb, depth, _ids, separate = camera(scene, actual, actual_yaw, None)
            down_depth = None
            head_sweep = []
            if arm == "separate_range":
                packet = separate
            elif arm == "upright_rgbd":
                packet = projected_depth_fan(depth, ANGLES)
            elif arm in ("dual_rgbd", "dual_rgbd_active_pan"):
                eye = np.array([actual[0], scene.floor_y + .42, actual[1]])
                heading = math.radians(actual_yaw)
                direction = np.array([math.sin(heading), 0., math.cos(heading)])
                tilt = math.radians(45.)
                down_direction = direction * math.cos(tilt) + np.array([
                    0., -math.sin(tilt), 0.])
                _down_rgb, down_depth = scene.observe(eye, down_direction,
                    owner_xz=None)
                packet = dual_view_fan(depth, down_depth, ANGLES)
                if arm == "dual_rgbd_active_pan" and np.isnan(packet[:, FRONT]).any():
                    for pan_deg in (-20., 20.):
                        head_yaw = math.radians(actual_yaw + pan_deg)
                        head_direction = np.array([math.sin(head_yaw), 0.,
                                                   math.cos(head_yaw)])
                        pan_rgb, pan_depth = scene.observe(eye, head_direction,
                            owner_xz=None)
                        pan_down_direction = head_direction * math.cos(tilt) + \
                            np.array([0., -math.sin(tilt), 0.])
                        pan_down_rgb, pan_down_depth = scene.observe(eye,
                            pan_down_direction, owner_xz=None)
                        extra = dual_view_fan(pan_depth, pan_down_depth,
                                              ANGLES - pan_deg)
                        packet = np.fmin(packet, extra)
                        prefix = f"pan_{'left' if pan_deg < 0 else 'right'}_{frame:03d}"
                        for name, image in (("rgb", pan_rgb),
                                ("down_rgb", pan_down_rgb)):
                            Image.fromarray(image).save(output / f"{prefix}_{name}.png")
                        for name, image in (("depth", pan_depth),
                                ("down_depth", pan_down_depth)):
                            Image.fromarray(np.where(np.isfinite(image),
                                np.clip(image * 1000., 0., 65535.), 0).astype(
                                    np.uint16)).save(output / f"{prefix}_{name}.png")
                        head_sweep.append({"pan_deg": pan_deg,
                            "rgb": relative(output / f"{prefix}_rgb.png"),
                            "depth": relative(output / f"{prefix}_depth.png"),
                            "down_rgb": relative(output / f"{prefix}_down_rgb.png"),
                            "down_depth": relative(output / f"{prefix}_down_depth.png"),
                            "remaining_unknown_front_sectors": int(np.isnan(
                                packet[:, FRONT]).sum())})
                        if not np.isnan(packet[:, FRONT]).any():
                            break
            else:
                raise ValueError(arm)
            writer.append_data(rgb)
            Image.fromarray(np.where(np.isfinite(depth),
                np.clip(depth * 1000., 0., 65535.), 0).astype(np.uint16)).save(
                    output / f"depth_{frame:03d}.png")
            if down_depth is not None:
                Image.fromarray(np.where(np.isfinite(down_depth),
                    np.clip(down_depth * 1000., 0., 65535.), 0).astype(np.uint16)).save(
                        output / f"down_depth_{frame:03d}.png")
            kind, amount, reason, health = choose(actor, actor_yaw, target, packet)
            row = {"frame": frame, "capture_time_s": frame * .2,
                "depth": relative(output / f"depth_{frame:03d}.png"),
                "down_depth": relative(output / f"down_depth_{frame:03d}.png")
                    if down_depth is not None else None,
                "actor_xz_m": actor.tolist(), "actor_yaw_deg": actor_yaw,
                "evaluator_actual_xz_m": actual.tolist(),
                "evaluator_actual_yaw_deg": actual_yaw,
                "head_sweep": head_sweep,
                "sensor_packet_m": encode_range(packet), "sensor_health": health,
                "decision": kind, "amount": amount, "reason": reason,
                "evaluator_true_target_gap_m": float(np.linalg.norm(actual - target_true))}
            if kind == "hold":
                result = reason
            elif kind == "turn":
                actor_yaw += amount
                actual_yaw += amount
            else:
                heading = math.radians(actual_yaw)
                candidate = actual + amount * np.array([
                    math.sin(heading), math.cos(heading)])
                eye = np.array([candidate[0], scene.floor_y + .42, candidate[1]])
                fan = scene.range_fan(eye, actual_yaw, owner_xz=None,
                    angles_deg=ANGLES)
                finite = fan[np.isfinite(fan)]
                accepted = bool(scene.floor_supported(*candidate) and len(finite)
                    and finite.min() >= base.BODY_RADIUS_M)
                row["evaluator_motion_accepted"] = accepted
                if accepted:
                    actual = candidate
                    actor += amount * np.array([
                        math.sin(math.radians(actor_yaw)),
                        math.cos(math.radians(actor_yaw))])
                else:
                    result = "simulator_rejected_motion"
            actions.append(row)
            if kind == "hold" or result == "simulator_rejected_motion":
                break
    baseline = json.loads((v2.ASSETS / key / "trace.json").read_text())
    last_base = baseline["actions"][-1]
    exact_baseline_endpoint = bool(result == "observed_target_standoff" and
        np.allclose(actual, last_base["evaluator_actual_xz_m"], atol=1e-6) and
        np.allclose(actor, last_base["actor_pose_world_xz_m"], atol=1e-6) and
        abs(actual_yaw - last_base["evaluator_actual_heading_deg"]) <= 1e-6 and
        abs(actor_yaw - last_base["actor_heading_deg"]) <= 1e-6)
    reconfirm = (baseline["reconfirmation_action"] if exact_baseline_endpoint
                 else None)
    trace = {"schema": "ape.item-approach-sensor-arm.v1", "scene_key": key,
        "arm": arm, "source_confirmed_stage": confirmed["stage"]["trace"],
        "observed_target_xz_m": target.tolist(),
        "policy": {"standoff_m": base.STANDOFF_M,
            "stop_tolerance_m": v2.TOLERANCE_M, "max_step_m": base.STEP_M,
            "max_turn_deg": base.TURN_DEG,
            "front_clearance": "both planes, all angles within +/-15 degrees known; unknown stops"},
        "result": result, "actions": actions,
        "exact_baseline_endpoint": exact_baseline_endpoint,
        "reused_fresh_owner_confirmation": reconfirm,
        "baseline_reconfirmation_trace": relative(v2.ASSETS / key / "reconfirm/episode_trace.json")
            if exact_baseline_endpoint else None,
        "video": relative(output / "first_person.mp4"),
        "final_actual_target_gap_m": float(np.linalg.norm(actual - target_true)),
        "limits": "Static development-exposed room and object. Ideal rendered sensors, yaw and commanded-pose kinematic proxy. Projected RGB-D coverage is a simulation adapter, not a physical device. No real IMU, joint force/contact or articulated body."}
    (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": key, "capture_group": case["capture_group"],
        "arm": arm, "result": result,
        "exact_baseline_endpoint": exact_baseline_endpoint,
        "owner_reconfirmed": reconfirm == "owner_confirmed_visual_identity",
        "final_actual_target_gap_m": trace["final_actual_target_gap_m"],
        "accepted_forward_steps": sum(x["decision"] == "forward" and
            x.get("evaluator_motion_accepted") for x in actions),
        "trace": relative(output / "trace.json"), "video": trace["video"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = json.loads(COHORT.read_text())["cases"]
    frozen = {"schema": "ape.item-approach-sensor-20-precommit.v1",
        "cohort": relative(COHORT), "cohort_sha256": sha(COHORT),
        "baseline_audit": relative(BASELINE_AUDIT),
        "baseline_audit_sha256": sha(BASELINE_AUDIT),
        "screens": [{"scene_key": c["scene_key"], "screen": c["screen"],
                    "sha256": sha(ROOT / c["screen"])} for c in cases],
        "arms": list(ARMS),
        "shared_policy": "Previously confirmed RGB-D item target, commanded pose/yaw, <=15deg turns, <=4cm steps, 55cm observed standoff with 1cm command-resolution band; both low and mid forward +/-15deg sectors must be known and free before step. No alternate route; unknown clearance means hold.",
        "score": ["reached_observed_standoff", "exact_baseline_endpoint",
                  "reused_fresh_owner_confirmation", "accepted_motion", "unknown_stops",
                  "video_depth_trace_alignment"],
        "limits": "Ideal static real-capture mesh sensors; depth projection adapter and same optical center for down view; no measured IMU/joint/contact or 50cm body dynamics."}
    if "dual_rgbd_active_pan" in ARMS:
        frozen["active_pan"] = ("If the fixed two-view front packet contains unknown sectors, "
            "take one head-only -20 degree pan, then +20 degree pan if still unknown, "
            "at the same body pose. Each pan takes upright and 45-degree-down "
            "RGB-D; reproject onto body-forward sectors and fuse only observed "
            "clearance/obstacles. Body holds if any required sector stays unknown.")
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen sensor comparison changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    if len(cases) != 20 or len({c["capture_group"] for c in cases}) != 20:
        raise ValueError("Expected 20 distinct captures")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cases:
        for arm in ARMS:
            path = OUT / case["scene_key"] / arm / "trace.json"
            if not path.exists():
                if args.max_new and launched >= args.max_new:
                    break
                row = run_arm(case, arm)
                launched += 1
            else:
                trace = json.loads(path.read_text())
                row = {"scene_key": case["scene_key"],
                    "capture_group": case["capture_group"], "arm": arm,
                    "result": trace["result"],
                    "exact_baseline_endpoint": trace["exact_baseline_endpoint"],
                    "owner_reconfirmed": trace["reused_fresh_owner_confirmation"] ==
                        "owner_confirmed_visual_identity",
                    "final_actual_target_gap_m": trace["final_actual_target_gap_m"],
                    "accepted_forward_steps": sum(x["decision"] == "forward" and
                        x.get("evaluator_motion_accepted") for x in trace["actions"]),
                    "trace": relative(path), "video": trace["video"]}
            rows.append(row)
            REPORT.write_text(json.dumps({"schema": "ape.item-approach-sensor-20-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows),
                "rows": rows}, indent=2) + "\n")
            print(row["scene_key"], arm, row["result"],
                  row["accepted_forward_steps"], flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
