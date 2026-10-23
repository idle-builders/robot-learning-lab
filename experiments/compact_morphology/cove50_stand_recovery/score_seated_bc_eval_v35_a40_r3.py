"""Source-verify V35 paired native replays against the unchanged V25 physical gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

from experiments.compact_morphology.cove50_stand_recovery import score_seated_reference_cuda_v26 as v26


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r3"
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ROUTES = ("bc", "model25", "scripted")
SEEDS = tuple(range(35511, 35519))


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
    report_path = ART / "seed35500-sentinel.json"
    native_path = ART / "seed35500-sentinel.npz"
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


def decode_video_frame(path: Path, plan: dict) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("V35 native MP4 missing or empty")
    with imageio.get_reader(str(path), format="ffmpeg") as reader:
        frame = reader.get_data(0)
    shape = list(frame.shape)
    pixel_std = float(np.std(frame))
    need(shape == plan["render_preflight_frame_shape"] and
         np.issubdtype(frame.dtype, np.integer) and
         np.isfinite(pixel_std) and
         pixel_std >= plan["render_preflight_min_pixel_std"],
         "V35 native MP4 did not decode to a nonblank 640x480 RGB frame")
    return {"decoded_frame_shape": shape, "decoded_first_frame_pixel_std": pixel_std,
            "encoded_video_bytes": path.stat().st_size}


def verify_render_preflight(plan: dict, ledger: dict) -> dict:
    need(ledger["render_preflight_status"] == "passed_unscored" and
         ledger["render_preflight_attempts"] == 1,
         "V35 paired one-step render preflight did not pass once")
    stem = ART / "seed35500-render-preflight"
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
    no_video_report = json.loads((ART / "seed35500-sentinel.json").read_text())
    need(report["initial_physics"] == no_video_report["initial_physics"] and
         report["initial_motors"] == no_video_report["initial_motors"],
         "V35 render toggled the initial sampled physics or motors")
    with np.load(native_path, allow_pickle=False) as rendered, np.load(
            ART / "seed35500-sentinel.npz", allow_pickle=False) as plain:
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
    stem = ART / "seed35002-timing-bc"
    report_path, native_path, video_path = (stem.with_suffix(ext) for ext in
                                            (".json", ".npz", ".mp4"))
    need(sha(report_path) == ledger["timing_report_sha256"] and
         sha(native_path) == ledger["timing_native_sha256"] and
         sha(video_path) == ledger["timing_video_sha256"],
         "V35 A40 timing report, trace or inline video differs")
    decode_video_frame(video_path, plan)
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
         not report["ended_early"] and not report["end_termination_terms"] and
         trace["actor_observation"].shape == (500, 73) and
         trace["raw_action"].shape == (500, 21) and
         trace["applied_action"].shape == (500, 21) and
         trace["processed_target"].shape == (500, 21) and
         np.isfinite(trace["actor_observation"]).all() and
         np.isfinite(trace["raw_action"]).all() and
         np.array_equal(trace["applied_action"], trace["raw_action"]) and
         not trace["selected_action_is_script"].any(),
         "V35 A40 timing source, hardware or full native route differs")
    checkpoint = ROOT / plan["source_paths"]["bc_checkpoint"]
    need(np.max(np.abs(actor_first_action(trace["actor_observation"][0], checkpoint) -
                       trace["raw_action"][0])) <= plan["first_action_parity_max_abs"],
         "V35 A40 timing first BC action differs")
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


def score_seed(seed: int, plan: dict, ledger: dict, model: mujoco.MjModel, gate: dict) -> dict:
    v26.ART = ART
    family = next(item for item in ledger["families"] if item["seed"] == seed)
    rows, reports, traces = {}, {}, {}
    for route in ROUTES:
        entry = family["routes"][route]
        need(entry["status"] == "completed", f"{seed} {route}: original incomplete")
        report_path = ART / f"seed{seed}-{route}.json"
        native_path = ART / f"seed{seed}-{route}.npz"
        video_path = ART / f"seed{seed}-{route}.mp4"
        need(sha(report_path) == entry["report_sha256"] and
             sha(native_path) == entry["native_sha256"] and
             sha(video_path) == entry["video_sha256"],
             f"{seed} {route}: report, trace or original video SHA differs")
        report = json.loads(report_path.read_text())
        with np.load(native_path, allow_pickle=False) as source:
            trace = {name: source[name].copy() for name in source.files}
        n = len(report["timeseries"])
        checkpoint_key = "bc_checkpoint" if route == "bc" else "v25_checkpoint"
        checkpoint = ROOT / plan["source_paths"][checkpoint_key]
        need(report["seed"] == seed and report["intervention_route"] == route and
             report["preflight_sentinel"] is False and
             report["render_preflight"] is False and
             report["mujoco_gl"] == plan["evaluator_env"]["MUJOCO_GL"] and
             report["nvidia_driver_capabilities"] == plan["evaluator_env"]["NVIDIA_DRIVER_CAPABILITIES"] and
             report["timing_probe"] is False and report["device"] == "cuda:0" and
             report["gpu_id"] == plan["gpu_id"] and
             report["nvidia_smi_name"] == plan["nvidia_smi_name"] and
             report["cuda_device_name"] == plan["cuda_device_name"] and
             report["hardware_shift"] == "fresh_A40_development_no_RTX4090_pooling" and
             report["body_sha256"] == plan["source_sha256"]["body"] and
             report["checkpoint_sha256"] == plan["source_sha256"][checkpoint_key] and
             tuple(PurePosixPath(report["checkpoint"]).parts[-3:]) ==
                tuple(PurePosixPath(plan["source_paths"][checkpoint_key]).parts[-3:]) and
             report["evaluator_source_sha256"] == plan["source_sha256"]["evaluator"] and
             report["evaluation_common_step_counter_after_load"] ==
                plan["evaluation_common_step_counter"] and
             report["actor_observation_terms"] == ["base_ang_vel", "projected_gravity",
                "joint_pos", "joint_vel", "actions", "command", "rise_phase"] and
             report["actor_observation_corruption"] is True and
             report["control_dt_s"] == 0.02 and 1 <= n <= 500 and
             ((n == 500 and not report["ended_early"]) or
              (n < 500 and report["ended_early"] and bool(report["end_termination_terms"]))) and
             trace["actor_observation"].shape == (n, 73) and
             trace["critic_observation"].shape == (n, 88) and
             trace["raw_action"].shape == (n, 21) and
             trace["applied_action"].shape == (n, 21) and
             trace["processed_target"].shape == (n, 21) and
             np.isfinite(trace["actor_observation"]).all() and
             np.isfinite(trace["raw_action"]).all(),
             f"{seed} {route}: source or native interface differs")
        need(np.allclose(trace["pre_action_time_s"], np.arange(n) * 0.02, atol=1e-8) and
             np.allclose(trace["actor_observation"][:, 72], trace["rise_phase_pre"], atol=1e-6) and
             np.array_equal(trace["previous_raw_action"][1:], trace["raw_action"][:-1]) and
             hashlib.sha256(trace["raw_action"][0].tobytes()).hexdigest() ==
                report["first_action_sha256"],
             f"{seed} {route}: pre-action/action history differs")
        selected = trace["selected_action_is_script"].astype(bool)
        expected = np.ones(n, dtype=bool) if route == "scripted" else np.zeros(n, dtype=bool)
        need(np.array_equal(selected, expected) and
             np.array_equal(trace["raw_action"],
                            trace["script_candidate_action"] if route == "scripted"
                            else trace["actor_candidate_action"]),
             f"{seed} {route}: executed action differs from declared route")
        if route != "scripted":
            actor_gap = float(np.max(np.abs(actor_first_action(
                trace["actor_observation"][0], checkpoint) - trace["raw_action"][0])))
            need(actor_gap <= 1e-5, f"{seed} {route}: saved actor first action gap {actor_gap}")
        else:
            actor_gap = None
        scale = np.asarray(report["action_process_scale"])
        offset = np.asarray(report["action_process_offset"])
        clip = report["action_process_clip"]
        need(scale.shape == (21,) and offset.shape == (21,) and
             report["wrapper_action_clip"] is None and
             np.array_equal(trace["applied_action"], trace["raw_action"]),
             f"{seed} {route}: action boundary differs")
        target = trace["applied_action"] * scale + offset
        if clip is not None:
            bounds = np.asarray(clip)
            need(bounds.shape == (21, 2), f"{seed} {route}: target clip shape differs")
            target = np.clip(target, bounds[:, 0], bounds[:, 1])
        target_gap = float(np.max(np.abs(target - trace["processed_target"])))
        need(target_gap <= 2e-6, f"{seed} {route}: processed target gap {target_gap}")
        row, _ = v26.score(seed, route, model, gate, plan["source_sha256"]["body"])
        row["saved_actor_first_action_max_abs_gap"] = actor_gap
        row["processed_target_max_abs_gap"] = target_gap
        row["original_video"] = str(video_path.relative_to(ROOT))
        row["original_video_sha256"] = sha(video_path)
        rows[route], reports[route], traces[route] = row, report, trace

    identity = ("initial_state_sha256", "initial_qpos", "initial_qvel",
                "initial_predicted_sole_gap_m", "initial_physics", "initial_motors")
    need(all(reports["bc"][key] == reports[route][key]
             for route in ROUTES[1:] for key in identity) and
         all(np.array_equal(traces["bc"]["actor_observation"][0],
                            traces[route]["actor_observation"][0]) and
             np.array_equal(traces["bc"]["critic_observation"][0],
                            traces[route]["critic_observation"][0])
             for route in ROUTES[1:]),
         f"{seed}: paired first observation, state, physics or motor draws differ")
    return {"seed": seed, "status": "complete_source_checked_three_route_family",
            "first_observation_and_full_physics_equal": True,
            "bc": rows["bc"], "model25": rows["model25"], "scripted": rows["scripted"],
            "bc_paired_win": rows["bc"]["full_gate"] and not rows["model25"]["full_gate"],
            "bc_regression_vs_model25": rows["model25"]["full_gate"] and
                                        not rows["bc"]["full_gate"]}


def main() -> None:
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    ledger_path = ART / "attempt-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    need(plan["source_sha256"]["scorer"] == sha(Path(__file__)) and
         all(sha(ROOT / rel) == plan["source_sha256"][name]
             for name, rel in plan["source_paths"].items()) and
         ledger["plan_sha256"] == sha(plan_path) and
         ledger["seed_families"] == list(SEEDS) and ledger["routes"] == list(ROUTES),
         "V35 plan, source or ledger changed")
    if ledger["sentinel_status"] != "not_attempted_unscored":
        verify_sentinel(plan, ledger)
    if ledger["render_preflight_status"] != "not_attempted_unscored":
        verify_render_preflight(plan, ledger)
    if ledger["timing_status"] != "not_attempted_unscored":
        verify_timing(plan, ledger)
    if any(entry["attempts"] for family in ledger["families"]
           for entry in family["routes"].values()):
        need(ledger["timing_status"] == "passed_unscored" and
             ledger["render_preflight_status"] == "passed_unscored",
             "V35 A40 scored original lacks prior full-video throughput gate")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    gate = json.loads((V25 / "plan.json").read_text())["physical_gate"]
    completed, incomplete = [], []
    for family in ledger["families"]:
        if all(family["routes"][route]["status"] == "completed" for route in ROUTES):
            completed.append(score_seed(family["seed"], plan, ledger, model, gate))
        else:
            incomplete.append({"seed": family["seed"],
                               "routes": {route: family["routes"][route]["status"]
                                          for route in ROUTES}})
    report = {"schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r3.score",
              "status": "complete" if (len(completed) == len(SEEDS) and
                                        ledger["sentinel_status"] == "passed_unscored" and
                                        ledger["render_preflight_status"] == "passed_unscored" and
                                        ledger["timing_status"] == "passed_unscored") else "partial",
        "sentinel_status": ledger["sentinel_status"],
        "render_preflight_status": ledger["render_preflight_status"],
              "timing_status": ledger["timing_status"],
              "plan_sha256": sha(plan_path), "attempt_ledger_sha256": sha(ledger_path),
              "hardware_shift": "fresh_A40_development_no_RTX4090_pooling",
              "source_checked_complete_families": len(completed),
              "planned_families": len(SEEDS), "incomplete_families": incomplete,
              "native_visits": sum(entry["attempts"] for family in ledger["families"]
                                   for entry in family["routes"].values()),
              "bc_full_gate": sum(row["bc"]["full_gate"] for row in completed),
              "model25_full_gate": sum(row["model25"]["full_gate"] for row in completed),
              "scripted_full_gate": sum(row["scripted"]["full_gate"] for row in completed),
              "bc_paired_wins": sum(row["bc_paired_win"] for row in completed),
              "bc_regressions": sum(row["bc_regression_vs_model25"] for row in completed),
              "rows": completed,
              "claim_limit": "Fresh development replay only; sealed 12-family score reserve unopened"}
    output = ART / "score.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({key: report[key] for key in
                      ("status", "source_checked_complete_families", "bc_full_gate",
                       "model25_full_gate", "scripted_full_gate", "bc_paired_wins")}, indent=2))


if __name__ == "__main__":
    main()
