"""V145p: native reverse-progress packet gate before a scripted owner show.

The two retained 50 Hz backward rollouts are historical Direct Frame data.
Their first 1.2 s of motion is transferred to the owner room only as a pose
counterfactual; neither native controller executes in that furnished room.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from PIL import Image

from articulated_person_scene import add_person
from audit_direct_structural_entry_v271 import V149, V150
import direct_structural_rooms_v270 as structural
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import infer
import run_pre_show_reserve_screen_v100 as v100
import validate_early_meeting_v128p as v128p


sys.path.insert(0, str(ROOT / "experiments/ape_high_level/sensors"))
from native_proprioceptive_odometry import BODY, SOURCE, replay  # noqa: E402


OUT = ROOT / "artifacts/ape-closed-loop-cove50-reverse-progress-gate"
SCRIPT = Path(__file__)
NATIVE_CASES = (
    "backward-varied300-backward-seed144",
    "planar200v3-backward",
)
SEED = 170
COMMAND_M = 0.12
ARRIVAL_TOLERANCE_M = 0.02
ARRIVAL_THRESHOLD_M = COMMAND_M - ARRIVAL_TOLERANCE_M
DECISION_FRAME = 60  # 1.2 s at the native 50 Hz sample rate.
SHOULDER_RAD = 0.6


def yaw(quaternion: np.ndarray) -> float:
    return 2 * math.atan2(float(quaternion[3]), float(quaternion[0]))


def local_delta(native_qpos: np.ndarray, frame: int) -> np.ndarray:
    """Rotate native world translation into its initial heading frame."""
    delta = native_qpos[frame, :2] - native_qpos[0, :2]
    angle = -yaw(native_qpos[0, 3:7])
    c, s = math.cos(angle), math.sin(angle)
    return np.array((c * delta[0] - s * delta[1],
                     s * delta[0] + c * delta[1]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Current mechanical-selection scope changed")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    v98_report = v128p.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_report.read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and row["case"][0] == SEED)
    staging_path = (v128p.V98 / "moving" / f"seed{SEED}-present" /
                    f"seed{SEED}-present/side-probe-00.npz")
    with np.load(staging_path) as packet:
        staging_qpos = packet["actor_qpos"].copy()
        staging_estimate = packet["estimated_pose_xyyaw"].copy()
    shoulder = model.joint("left_shoulder_pitch").qposadr[0]
    if model.nq != len(staging_qpos):
        raise AssertionError("Native and owner-room qpos mismatch")
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"][
        "hue_histogram"])

    old_model = structural.structural_model
    old_out = v100.OUT
    old_sequences = v100.SEQUENCES.copy()
    old_directions = v100.DIRECTIONS.copy()
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v100.SEQUENCES["wait"] = ("wait",)
    v100.DIRECTIONS["wait"] = (0., 0.)
    rows = []
    try:
        for name in NATIVE_CASES:
            native_path = SOURCE / f"{name}.npz"
            native_report = native_path.with_suffix(".json")
            with np.load(native_path, allow_pickle=False) as saved:
                native_qpos = saved["qpos"].astype(float)
                dt = float(saved["dt"])
            if abs(DECISION_FRAME * dt - 1.2) > 1e-9:
                raise AssertionError("Native packet time changed")
            odometry, samples = replay(model, native_path)
            actual_delta = local_delta(native_qpos, DECISION_FRAME)
            actual_final = local_delta(native_qpos, len(native_qpos) - 1)
            packet_deltas = {}
            for estimator in (
                "ideal_orientation_joint_contact",
                "ideal_orientation_joint_all_feet",
                "combined_illustrative_impairments",
            ):
                estimate = np.asarray(samples[DECISION_FRAME][
                    "estimated_xy_m"][estimator])
                estimate_0 = np.asarray(samples[0]["estimated_xy_m"][estimator])
                native_yaw = yaw(native_qpos[0, 3:7])
                c, s = math.cos(native_yaw), math.sin(native_yaw)
                delta = estimate - estimate_0
                local = np.array((c * delta[0] + s * delta[1],
                                  -s * delta[0] + c * delta[1]))
                packet_deltas[estimator] = {
                    "estimated_local_xy_m": local.round(5).tolist(),
                    "estimated_backward_progress_m": round(-float(local[0]), 5),
                    "progress_gate_reached": bool(-local[0] >= ARRIVAL_THRESHOLD_M),
                }
            actual_qpos = staging_qpos.copy()
            actual_qpos[:2] += actual_delta
            actual_qpos[shoulder] = SHOULDER_RAD
            station_estimate = staging_estimate.copy()
            station_estimate[:2] += np.asarray(packet_deltas[
                "ideal_orientation_joint_contact"]["estimated_local_xy_m"])
            scene = {}
            for case in ("clear", "hand_close"):
                v100.OUT = OUT / name / case
                scene[case] = v100.run_one(SEED, case, "wait", actual_qpos,
                                           station_estimate,
                                           tuple(prior["case"]))
                image_meta = scene[case]["owner_image"]
                image_path = ROOT / image_meta["actor_frame_path"]
                if sha(image_path) != image_meta["actor_frame_sha256"]:
                    raise AssertionError("Owner RGB-D packet changed")
                with np.load(image_path) as frame:
                    rgb = Image.fromarray(frame["rgb"].copy())
                    depth = frame["depth_m"].copy()
                    pose = (frame["camera_origin_m"].copy(),
                            frame["camera_rotation"].copy())
                depth[(depth < .08) | (depth > 1.5)] = np.nan
                started = time.perf_counter()
                choice = infer(vision, exemplar, owner_hist,
                               rgb, depth, pose, None, None)
                box = image_meta["evaluator_truth_box"]
                overlap = (iou(choice["accepted_box"], box)
                           if choice["accepted_box"] is not None
                           and box is not None else 0.)
                scene[case]["generic_taught_item_accepted"] = bool(
                    choice["accepted_box"] is not None)
                scene[case]["evaluator_item_box_iou"] = round(
                    float(overlap), 5)
                scene[case]["local_inference_wall_s"] = round(
                    time.perf_counter() - started, 4)
            rows.append({
                "native_case": name,
                "native_trace": str(native_path.relative_to(ROOT)),
                "native_trace_sha256": sha(native_path),
                "native_report": str(native_report.relative_to(ROOT)),
                "native_report_sha256": sha(native_report),
                "odometry": odometry,
                "packet_time_s": round(DECISION_FRAME * dt, 4),
                "actual_local_xy_m_evaluator_only": actual_delta.round(5).tolist(),
                "actual_backward_progress_m_evaluator_only": round(
                    -float(actual_delta[0]), 5),
                "actual_final_local_xy_m_evaluator_only": actual_final.round(5).tolist(),
                "packet_progress": packet_deltas,
                "command_clock_assumed_arrival": True,
                "owner_show_if_command_clock_assumes_arrival": scene,
            })
            print(name, "actual retreat", round(-float(actual_delta[0]), 4),
                  "packet retreat", packet_deltas[
                      "ideal_orientation_joint_contact"][
                          "estimated_backward_progress_m"],
                  "show gaps", {k: v["minimum_scene_gap_m"]
                                for k, v in scene.items()}, flush=True)
    finally:
        structural.structural_model = old_model
        v100.OUT = old_out
        v100.SEQUENCES.clear()
        v100.SEQUENCES.update(old_sequences)
        v100.DIRECTIONS.clear()
        v100.DIRECTIONS.update(old_directions)

    report = {
        "schema": "robot-learning-lab.ape-reverse-meeting-progress-gate.v1",
        "source_sha256": sha(SCRIPT),
        "historical_body": str(BODY.relative_to(ROOT)),
        "historical_body_sha256": sha(BODY),
        "current_design_status": pointer["status"],
        "staging_packet": str(staging_path.relative_to(ROOT)),
        "staging_packet_sha256": sha(staging_path),
        "v98_report_sha256": sha(v98_report),
        "generic_exemplar_sha256": sha(V149),
        "owner_color_reference_sha256": sha(V150),
        "requested_backward_m": COMMAND_M,
        "progress_arrival_threshold_m": ARRIVAL_THRESHOLD_M,
        "scope": "Two retained historical Direct Frame backward-policy traces are "
                 "replayed with idealized simulated orientation/joints/foot "
                 "contact, an all-feet ablation, and illustrative packet "
                 "impairments. Native root is evaluator-only. At 1.2s their "
                 "actual root offset is transferred to the furnished owner "
                 "room and held in a raised-arm kinematic posture for the "
                 "owner-show counterfactual with a rendered generic RGB-D "
                 "taught-item gate. Neither policy executes there; "
                 "the camera, route map and moving prop are not fed into the "
                 "native policy. This is a progress-handoff test, not a "
                 "qualified reverse skill or sensor specification.",
        "rows": rows,
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
