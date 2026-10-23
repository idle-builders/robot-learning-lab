"""Source-verify unscored A40-r5 EGL, video and timing readiness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio



ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r5"
ROUTES = ()
SEEDS = ()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(detail)


def actor_first_action(obs: np.ndarray, checkpoint: Path) -> np.ndarray:
    actor = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
    x = torch.from_numpy(obs.copy()).float().unsqueeze(0)
    x = (x - actor["obs_normalizer._mean"]) / (actor["obs_normalizer._std"] + 0.01)
    for layer in (0, 2, 4, 6):
        x = F.linear(x, actor[f"mlp.{layer}.weight"], actor[f"mlp.{layer}.bias"])
        if layer != 6:
            x = F.elu(x)
    return x[0].detach().numpy()


def verify_sentinel(plan: dict, ledger: dict) -> None:
    need(ledger["sentinel_status"] == "passed_unscored" and
         ledger["sentinel_attempts"] == 1,
         "V35 exact CUDA BC sentinel did not pass once")
    report_path = ART / "seed35900-sentinel.json"
    native_path = ART / "seed35900-sentinel.npz"
    need(sha(report_path) == ledger["sentinel_report_sha256"] and
         sha(native_path) == ledger["sentinel_native_sha256"],
         "V35 sentinel source SHA differs from original")
    report = json.loads(report_path.read_text())
    with np.load(native_path, allow_pickle=False) as source:
        actor = source["actor_observation"].copy()
        action = source["raw_action"].copy()
        applied = source["applied_action"].copy()
        target = source["processed_target"].copy()
    need(report["seed"] == plan["sentinel_seed"] and
         report["preflight_sentinel"] is True and
         report["render_preflight"] is False and
         report["timing_probe"] is False and
         report["mujoco_gl"] == plan["evaluator_env"]["MUJOCO_GL"] and
         report["nvidia_driver_capabilities"] == plan["evaluator_env"]["NVIDIA_DRIVER_CAPABILITIES"] and
         report["intervention_route"] == "bc" and
         report["device"] == "cuda:0" and
         report["gpu_id"] == plan["gpu_id"] and
         report["nvidia_smi_name"] == plan["nvidia_smi_name"] and
         report["cuda_device_name"] == plan["cuda_device_name"] and
         report["hardware_shift"] == "fresh_A40_development_no_RTX4090_pooling" and
         report["body_sha256"] == plan["source_sha256"]["body"] and
         report["checkpoint_sha256"] == plan["source_sha256"]["bc_checkpoint"] and
         report["evaluator_source_sha256"] == plan["source_sha256"]["evaluator"] and
         report["evaluation_common_step_counter_after_load"] ==
             plan["evaluation_common_step_counter"] and
         report["duration_s"] == 0.02 and len(report["timeseries"]) == 1 and
         actor.shape == (1, 73) and action.shape == (1, 21) and
         applied.shape == (1, 21) and target.shape == (1, 21) and
         np.isfinite(actor).all() and np.isfinite(action).all() and
         np.array_equal(applied, action),
         "V35 CUDA sentinel source or first interface differs")
    checkpoint = ROOT / plan["source_paths"]["bc_checkpoint"]
    need(np.max(np.abs(actor_first_action(actor[0], checkpoint) - action[0])) <=
         plan["first_action_parity_max_abs"],
         "V35 CUDA sentinel first saved actor action differs")
    scale = np.asarray(report["action_process_scale"])
    offset = np.asarray(report["action_process_offset"])
    reconstructed = applied[0] * scale + offset
    clip = report["action_process_clip"]
    if clip is not None:
        bounds = np.asarray(clip)
        need(bounds.shape == (21, 2), "V35 CUDA sentinel target clip shape differs")
        reconstructed = np.clip(reconstructed, bounds[:, 0], bounds[:, 1])
    need(scale.shape == (21,) and offset.shape == (21,) and
         report["wrapper_action_clip"] is None and
         np.max(np.abs(reconstructed - target[0])) <=
             plan["processed_target_parity_max_abs"],
         "V35 CUDA sentinel first processed target differs")


def decode_video_frame(path: Path, plan: dict, *, expected_frames: int = 1) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("V35 native MP4 missing or empty")
    count = 0
    first_std = None
    minimum_std = float("inf")
    with imageio.get_reader(str(path), format="ffmpeg") as reader:
        for frame in reader:
            shape = list(frame.shape)
            pixel_std = float(np.std(frame))
            need(shape == plan["render_preflight_frame_shape"] and
                 np.issubdtype(frame.dtype, np.integer) and
                 np.isfinite(pixel_std) and
                 pixel_std >= plan["render_preflight_min_pixel_std"],
                 f"V35 native MP4 frame {count} is not nonblank 640x480 RGB")
            if first_std is None:
                first_std = pixel_std
            minimum_std = min(minimum_std, pixel_std)
            count += 1
            if count > expected_frames:
                raise ValueError("V35 native MP4 contains more frames than frozen route")
    need(count == expected_frames, "V35 native MP4 frame count differs from frozen route")
    return {"decoded_frame_shape": plan["render_preflight_frame_shape"],
            "decoded_frame_count": count,
            "decoded_first_frame_pixel_std": first_std,
            "decoded_minimum_frame_pixel_std": minimum_std,
            "encoded_video_bytes": path.stat().st_size}


def verify_render_preflight(plan: dict, ledger: dict) -> dict:
    need(ledger["render_preflight_status"] == "passed_unscored" and
         ledger["render_preflight_attempts"] == 1,
         "V35 paired one-step render preflight did not pass once")
    stem = ART / "seed35900-render-preflight"
    report_path, native_path, video_path = (stem.with_suffix(ext) for ext in
                                            (".json", ".npz", ".mp4"))
    need(sha(report_path) == ledger["render_preflight_report_sha256"] and
         sha(native_path) == ledger["render_preflight_native_sha256"] and
         sha(video_path) == ledger["render_preflight_video_sha256"],
         "V35 render preflight report, trace or video source differs")
    report = json.loads(report_path.read_text())
    need(report["seed"] == plan["render_preflight_seed"] and
         report["preflight_sentinel"] is True and
         report["render_preflight"] is True and
         report["timing_probe"] is False and
         report["device"] == "cuda:0" and
         report["mujoco_gl"] == plan["evaluator_env"]["MUJOCO_GL"] and
         report["nvidia_driver_capabilities"] == plan["evaluator_env"]["NVIDIA_DRIVER_CAPABILITIES"] and
         report["gpu_id"] == plan["gpu_id"] and
         report["body_sha256"] == plan["source_sha256"]["body"] and
         report["checkpoint_sha256"] == plan["source_sha256"]["bc_checkpoint"] and
         report["evaluator_source_sha256"] == plan["source_sha256"]["evaluator"] and
         report["duration_s"] == 0.02 and len(report["timeseries"]) == 1 and
         not report["ended_early"] and not report["end_termination_terms"],
         "V35 rendered one-step route differs")
    no_video_report = json.loads((ART / "seed35900-sentinel.json").read_text())
    need(report["initial_physics"] == no_video_report["initial_physics"] and
         report["initial_motors"] == no_video_report["initial_motors"],
         "V35 render toggled the initial sampled physics or motors")
    with np.load(native_path, allow_pickle=False) as rendered, np.load(
            ART / "seed35900-sentinel.npz", allow_pickle=False) as plain:
        gaps = {}
        for name in ("initial_qpos", "initial_qvel", "actor_observation",
                     "critic_observation", "raw_action", "applied_action",
                     "processed_target", "qpos", "qvel"):
            a, b = rendered[name], plain[name]
            need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                 f"V35 paired render source shape or finite values differ: {name}")
            gaps[name] = float(np.max(np.abs(a - b)))
    for name, gap in gaps.items():
        limit = (plan["render_preflight_physical_max_abs"]
                 if name in ("initial_qpos", "initial_qvel", "qpos", "qvel") else
                 plan["processed_target_parity_max_abs"] if name == "processed_target" else
                 plan["render_preflight_actor_max_abs"])
        need(gap <= limit, f"V35 render/no-render first-step parity failed: {name} gap={gap}")
    return {"first_step_max_abs_gaps": gaps, **decode_video_frame(video_path, plan)}


def verify_timing(plan: dict, ledger: dict) -> None:
    need(ledger["timing_status"] == "passed_unscored" and
         ledger["timing_attempts"] == 1 and
         ledger["timing_elapsed_wall_s"] <= plan["timing_max_elapsed_wall_s"],
         "V35 A40 full-video throughput probe did not pass once")
    stem = ART / "seed35003-timing-environment"
    report_path, native_path, video_path = (stem.with_suffix(ext) for ext in
                                            (".json", ".npz", ".mp4"))
    need(sha(report_path) == ledger["timing_report_sha256"] and
         sha(native_path) == ledger["timing_native_sha256"] and
         sha(video_path) == ledger["timing_video_sha256"],
         "V35 A40 timing report, trace or inline video differs")
    decoded = decode_video_frame(video_path, plan, expected_frames=250)
    report = json.loads(report_path.read_text())
    with np.load(native_path, allow_pickle=False) as source:
        trace = {name: source[name].copy() for name in source.files}
    need(report["seed"] == plan["timing_seed"] and
         report["intervention_route"] == plan["timing_route"] and
         report["timing_probe"] is True and
         report["preflight_sentinel"] is False and
         report["render_preflight"] is False and
         report["mujoco_gl"] == plan["evaluator_env"]["MUJOCO_GL"] and
         report["nvidia_driver_capabilities"] == plan["evaluator_env"]["NVIDIA_DRIVER_CAPABILITIES"] and
         report["device"] == "cuda:0" and
         report["gpu_id"] == plan["gpu_id"] and
         report["nvidia_smi_name"] == plan["nvidia_smi_name"] and
         report["cuda_device_name"] == plan["cuda_device_name"] and
         report["hardware_shift"] == "fresh_A40_development_no_RTX4090_pooling" and
         report["body_sha256"] == plan["source_sha256"]["body"] and
         report["checkpoint_sha256"] == plan["source_sha256"]["bc_checkpoint"] and
         report["evaluator_source_sha256"] == plan["source_sha256"]["evaluator"] and
         report["evaluation_common_step_counter_after_load"] ==
             plan["evaluation_common_step_counter"] and
         report["duration_s"] == 10.0 and len(report["timeseries"]) == 500 and
         report["timing_route_action"] == "zero" and
         report["control_step_count"] == 500 and
         report["render_frame_count"] == 250 and
         decoded["decoded_frame_count"] == 250 and
         not report["ended_early"] and
         trace["actor_observation"].shape == (500, 73) and
         trace["raw_action"].shape == (500, 21) and
         trace["applied_action"].shape == (500, 21) and
         trace["processed_target"].shape == (500, 21) and
         trace["episode_step_index"].shape == (500,) and
         trace["post_done"].shape == (500,) and
         np.isfinite(trace["actor_observation"]).all() and
         np.array_equal(trace["raw_action"], np.zeros_like(trace["raw_action"])) and
         np.array_equal(trace["applied_action"], trace["raw_action"]) and
         not trace["selected_action_is_script"].any(),
         "V35 A40 timing source, hardware or full native route differs")
    done_steps = [step + 1 for step, flag in enumerate(trace["post_done"]) if flag]
    reset_steps = [step for step in done_steps if step < 500]
    events = report["native_done_events"]
    resets = report["manual_reset_events"]
    need(report["native_done_count"] == len(done_steps) == len(events) and
         report["manual_reset_count"] == len(reset_steps) == len(resets) and
         [row["after_control_step"] for row in events] == done_steps and
         [row["after_control_step"] for row in resets] == reset_steps and
         all(row["done"] == bool(trace["post_done"][index]) for index, row in
             enumerate(report["timeseries"])) and
         all(resets[index]["termination_terms"] == events[index]["termination_terms"]
             for index in range(len(resets))) and
         (report["end_termination_terms"] == events[-1]["termination_terms"]
          if done_steps and done_steps[-1] == 500 else
          report["end_termination_terms"] is None),
         "V35 A40 timing done/reset ledger differs from 500 native steps")
    episode_step = 0
    for index in range(500):
        need(int(trace["episode_step_index"][index]) == episode_step and
             bool(trace["previous_action_valid"][index]) == (episode_step > 0),
             f"V35 A40 timing episode continuation differs at step {index}")
        episode_step = 0 if bool(trace["post_done"][index]) else episode_step + 1
    scale = np.asarray(report["action_process_scale"])
    offset = np.asarray(report["action_process_offset"])
    need(scale.shape == (21,) and offset.shape == (21,) and
         report["wrapper_action_clip"] is None,
         "V35 A40 timing action boundary differs")
    reconstructed = trace["applied_action"] * scale + offset
    if report["action_process_clip"] is not None:
        bounds = np.asarray(report["action_process_clip"])
        need(bounds.shape == (21, 2), "V35 A40 timing target clip shape differs")
        reconstructed = np.clip(reconstructed, bounds[:, 0], bounds[:, 1])
    need(np.max(np.abs(reconstructed - trace["processed_target"])) <=
         plan["processed_target_parity_max_abs"],
         "V35 A40 timing processed target differs")


def main() -> None:
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    ledger_path = ART / "attempt-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    need(plan["source_sha256"]["scorer"] == sha(Path(__file__)) and
         all(sha(ROOT / rel) == plan["source_sha256"][name]
             for name, rel in plan["source_paths"].items()) and
         ledger["plan_sha256"] == sha(plan_path) and
         ledger["seed_families"] == [] and ledger["routes"] == [] and
         ledger["families"] == [] and plan["original_visits_max"] == 0,
         "A40-r5 environment-only plan, source or ledger changed")
    if ledger["sentinel_status"] == "passed_unscored":
        verify_sentinel(plan, ledger)
    if ledger["render_preflight_status"] == "passed_unscored":
        verify_render_preflight(plan, ledger)
    if ledger["timing_status"] == "passed_unscored":
        verify_timing(plan, ledger)
    statuses = {name: ledger[name + "_status"] for name in
                ("sentinel", "render_preflight", "timing")}
    passed = all(value == "passed_unscored" for value in statuses.values())
    report = {
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r5.readiness",
        "status": "passed_environment_readiness" if passed else "incomplete_or_failed",
        "claim_limit": "Unscored exact-task environment readiness only; zero learned-control qualification",
        "plan_sha256": sha(plan_path),
        "attempt_ledger_sha256": sha(ledger_path),
        "diagnostic_statuses": statuses,
        "diagnostic_attempts": {name: ledger[name + "_attempts"] for name in statuses},
        "scored_visits": 0, "optimizer_steps": 0,
        "independent_score_reserve_unopened": plan["v35_independent_score_seeds_sealed"],
    }
    output = ART / "readiness.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
