"""Prepare a source-bound V35 phase-script BC seed after V34 is closed.

This is a data/CPU fitting entrypoint, not a CUDA score or a feedback expert.
No V34 artifact is changed. `inspect-v26` is a schema-only preflight on opened
historical episodes; it cannot produce a V35 bank or checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
V34 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-trigger-v34"
V35 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-v35"
V25_CHECKPOINT = (ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
                  / "checkpoints/model_25.pt")
EXPECTED_TERMS = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                  "actions", "command", "rise_phase"]
VALIDATION_SEEDS = tuple(range(35101, 35109))
SCORE_SEEDS = tuple(range(35201, 35213))
HELD_V25 = tuple(range(2240, 2252))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(detail)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def cpu_runtime() -> dict:
    import torch

    return {
        "execution_device": "cpu",
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "numpy_module": str(Path(np.__file__).resolve()),
        "torch": torch.__version__,
        "torch_module": str(Path(torch.__file__).resolve()),
        "torch_cuda_build": torch.version.cuda,
    }


def refuse_existing(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    need(not existing, "Refusing to overwrite existing V35 result: " + ", ".join(existing))


def inspect_v26() -> None:
    """Check only the old trace schema; never save its frames for V35 fitting."""
    import torch

    trace = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26/seed2231-scripted.npz"
    actor_trace = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26/seed2231-model25.npz"
    with np.load(trace, allow_pickle=False) as source:
        shapes = {name: list(source[name].shape) for name in source.files}
        need(shapes["actor_observation"] == [500, 73] and
             shapes["raw_action"] == [500, 21], "V26 old schema differs")
    saved = torch.load(V25_CHECKPOINT, map_location="cpu", weights_only=True)["actor_state_dict"]
    with np.load(actor_trace, allow_pickle=False) as source:
        obs = torch.from_numpy(source["actor_observation"].copy()).float()
        actions = source["raw_action"].copy()
    x = (obs - saved["obs_normalizer._mean"]) / (saved["obs_normalizer._std"] + 0.01)
    for layer in (0, 2, 4, 6):
        x = torch.nn.functional.linear(x, saved[f"mlp.{layer}.weight"],
                                       saved[f"mlp.{layer}.bias"])
        if layer != 6:
            x = torch.nn.functional.elu(x)
    action_gap = float(np.max(np.abs(x.detach().numpy() - actions)))
    need(action_gap < 1e-5, "Saved V25 actor action differs on opened V26 CUDA trace")
    print(json.dumps({"role": "schema_preflight_only", "source": str(trace.relative_to(ROOT)),
                      "source_sha256": sha(trace), "shapes": shapes,
                      "opened_actor_trace_sha256": sha(actor_trace),
                      "v25_actor_checkpoint_sha256": sha(V25_CHECKPOINT),
                      "opened_actor_action_max_abs_gap": action_gap}, indent=2))


def read_closed_v34(closure_path: Path) -> tuple[dict, dict, dict, dict]:
    closure = load_json(closure_path)
    plan_path, ledger_path, score_path = (V34 / name for name in
                                          ("plan.json", "attempt-ledger.json", "local-score.json"))
    guard_path = V34 / "fresh-pod-create-guard.json"
    plan, ledger, score = map(load_json, (plan_path, ledger_path, score_path))
    guard = load_json(guard_path)
    need(closure.get("schema") == "robot-learning-lab.cove50-v34-closed-for-v35.v1" and
         closure.get("v34_closed") is True and
         closure.get("no_more_v34_originals") is True and
         closure.get("closure_status") in ("complete", "stopped_partial") and
         isinstance(closure.get("closed_at_utc"), str) and
         closure.get("plan_sha256") == sha(plan_path) and
         closure.get("final_ledger_sha256") == sha(ledger_path) and
         closure.get("final_local_score_sha256") == sha(score_path) and
         closure.get("final_guard_state_sha256") == sha(guard_path) and
         guard.get("status") == "stopped" and
         guard.get("stop_requested") is True and
         str(guard.get("provider_runtime_status", "")).lower() not in
             ("running", "starting", "queued"),
         "V34 is not explicitly closed against these exact final files")
    need(ledger["plan_sha256"] == closure["plan_sha256"] and
         score["plan_sha256"] == closure["plan_sha256"] and
         score["attempt_ledger_sha256"] == closure["final_ledger_sha256"] and
         not score["pending_sync"] and not score["verification_failures"] and
         score["exact_cuda_sentinel_passed"] is True,
         "V34 source score/ledger/sentinel is incomplete or invalid")
    need(closure["closure_status"] != "complete" or
         score["source_checked_complete_families"] == len(plan["seed_families"]),
         "Complete closure requires every family source checked")
    return closure, plan, ledger, score


def accepted_episode(row: dict, ledger_family: dict, plan: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    seed = row["seed"]
    score = row["scripted"]
    entry = ledger_family["routes"]["scripted"]
    need(row["status"] == "complete_source_checked_four_route_family" and
         score["full_gate"] is True and not score["failure_reasons"] and
         all(score["physical_gate"].values()) and entry["status"] == "completed",
         f"seed {seed}: scripted source/gate not accepted")
    report_path = V34 / f"seed{seed}-scripted.json"
    native_path = V34 / f"seed{seed}-scripted.npz"
    need(sha(report_path) == entry["report_sha256"] == score["source_report_sha256"] and
         sha(native_path) == entry["native_sha256"] == score["native_trajectory_sha256"],
         f"seed {seed}: report/native hash mismatch")
    report = load_json(report_path)
    need(report["seed"] == seed and report["intervention_route"] == "scripted" and
         report["device"] == "cuda:0" and report["preflight_sentinel"] is False and
         report["duration_s"] == 10.0 and report["control_dt_s"] == 0.02 and
         report["body_sha256"] == plan["source_sha256"]["body"] and
         report["checkpoint_sha256"] == plan["source_sha256"]["checkpoint"] and
         report["evaluator_source_sha256"] == plan["source_sha256"]["evaluator"] and
         report["actor_observation_terms"] == EXPECTED_TERMS and
         report["actor_observation_corruption"] is True and
         len(report["timeseries"]) == 500,
         f"seed {seed}: native report contract differs")
    with np.load(native_path, allow_pickle=False) as trace:
        obs = trace["actor_observation"].copy()
        action = trace["raw_action"].copy()
        initial_hash = hashlib.sha256(trace["initial_qpos"].tobytes() +
                                      trace["initial_qvel"].tobytes()).hexdigest()
        need(obs.shape == (500, 73) and action.shape == (500, 21) and
             trace["critic_observation"].shape == (500, 88) and
             initial_hash == report["initial_state_sha256"] and
             hashlib.sha256(action[0].tobytes()).hexdigest() == report["first_action_sha256"] and
             np.all(trace["selected_action_is_script"]) and
             np.array_equal(action, trace["script_candidate_action"]) and
             np.allclose(trace["pre_action_time_s"], np.arange(500) * 0.02, atol=1e-8) and
             np.allclose(obs[:, 72], trace["rise_phase_pre"], atol=1e-6) and
             np.array_equal(trace["previous_raw_action"][1:], action[:-1]) and
             np.isfinite(obs).all() and np.isfinite(action).all(),
             f"seed {seed}: pre-action/action alignment or finite contract differs")
    friction = report["initial_physics"]["sole_slide_friction"]
    need(len(friction) == 2 and abs(friction[0] - friction[1]) < 1e-6,
         f"seed {seed}: sole friction draw differs")
    detail = {"seed": seed, "frames": 500,
              "report_sha256": sha(report_path), "native_sha256": sha(native_path),
              "initial_state_sha256": report["initial_state_sha256"],
              "predicted_sole_gap_m": report["initial_predicted_sole_gap_m"],
              "sole_friction": friction[0],
              "friction_stratum": "below_nominal" if friction[0] < 1.0 else "at_or_above_nominal",
              "rated_leg_fraction": score["rated_leg_fraction"]}
    return obs, action, detail


def build_bank(closure_path: Path) -> None:
    closure, plan, ledger, score = read_closed_v34(closure_path)
    bank = V35 / "bank.npz"
    manifest_path = V35 / "bank-manifest.json"
    refuse_existing(bank, manifest_path)
    need(sha(V25_CHECKPOINT) == plan["source_sha256"]["checkpoint"],
         "V25 actor checkpoint changed")
    score_rows = {row["seed"]: row for row in score["rows"]}
    ledger_rows = {row["seed"]: row for row in ledger["families"]}
    need(set(score_rows) <= set(plan["seed_families"]) and
         not (set(score_rows) & (set(VALIDATION_SEEDS) | set(SCORE_SEEDS) | set(HELD_V25))),
         "V34 train families overlap reserves")
    observations, actions, accepted, rejected = [], [], [], []
    for seed in plan["seed_families"]:
        row = score_rows.get(seed)
        if row is None:
            rejected.append({"seed": seed, "reason": "no_complete_source_checked_four_route_family"})
        elif not row["scripted"]["full_gate"]:
            rejected.append({"seed": seed, "reason": "scripted_failed_full_physical_gate",
                             "clauses": row["scripted"]["failure_reasons"]})
        else:
            obs, act, detail = accepted_episode(row, ledger_rows[seed], plan)
            observations.append(obs.astype(np.float32))
            actions.append(act.astype(np.float32))
            accepted.append(detail)
    need(len(accepted) >= 6 and len({r["friction_stratum"] for r in accepted}) >= 2,
         "Small BC seed needs >=6 valid whole episodes across both friction strata")
    V35.mkdir(parents=True, exist_ok=True)
    with bank.open("xb") as output:
        np.savez_compressed(output,
                            seed=np.asarray([r["seed"] for r in accepted], dtype=np.int32),
                            actor_observation=np.stack(observations),
                            raw_action=np.stack(actions))
    manifest = {
        "schema": "robot-learning-lab.cove50-seated-bc.v35.bank",
        "claim_limit": "On-route phase-script demonstration bank; no learned or closed-loop result",
        "preparer_source_sha256": sha(Path(__file__)),
        "fitter_source_sha256": sha(Path(__file__)),
        "cpu_runtime": cpu_runtime(),
        "v34_closure_sha256": sha(closure_path), "v34_plan_sha256": closure["plan_sha256"],
        "v34_closure_path": str(closure_path.resolve()),
        "v34_final_ledger_sha256": closure["final_ledger_sha256"],
        "v34_final_score_sha256": closure["final_local_score_sha256"],
        "v34_final_guard_state_sha256": closure["final_guard_state_sha256"],
        "v25_checkpoint_sha256": sha(V25_CHECKPOINT),
        "bank_sha256": sha(bank), "accepted": accepted, "rejected": rejected,
        "train_families": [row["seed"] for row in accepted],
        "internal_selection": "leave_one_whole_training_family_out, then refit on all accepted",
        "fresh_development_validation_seed_families": list(VALIDATION_SEEDS),
        "separate_unopened_score_seed_families": list(SCORE_SEEDS),
        "v25_historical_held_seeds_stay_sealed": list(HELD_V25),
        "input": "native pre-action corruption-enabled 73D actor vector, not 88D critic",
        "label": "matching 21D pre-clip executed scripted raw action",
        "control_dt_s": 0.02,
        "friction_stratum_boundary": 1.0,
        "friction_stratum_limit": "Descriptive observed split only; not a robust friction claim",
    }
    with manifest_path.open("x") as output:
        output.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"accepted_families": len(accepted), "rejected_families": len(rejected),
                      "training_frames": 500 * len(accepted),
                      "bank_sha256": manifest["bank_sha256"]}, indent=2))


def fit(bank_path: Path, manifest_path: Path, max_epochs: int, seed: int) -> None:
    """Small deterministic CPU feasibility fit; no native-control claim."""
    import torch
    from torch import nn

    need(1 <= max_epochs <= 30, "V35 diagnostic is capped at 30 epochs")
    checkpoint = bank_path.parent / "model_bc_seed.pt"
    fit_report_path = bank_path.parent / "fit-report.json"
    refuse_existing(checkpoint, fit_report_path)
    manifest = load_json(manifest_path)
    closure_path = Path(manifest.get("v34_closure_path", ""))
    need(manifest["schema"] == "robot-learning-lab.cove50-seated-bc.v35.bank" and
         manifest["preparer_source_sha256"] == sha(Path(__file__)) and
         manifest["fitter_source_sha256"] == sha(Path(__file__)) and
         manifest["cpu_runtime"] == cpu_runtime() and
         closure_path.is_file() and sha(closure_path) == manifest["v34_closure_sha256"] and
         sha(bank_path) == manifest["bank_sha256"] and
         sha(V25_CHECKPOINT) == manifest["v25_checkpoint_sha256"] and
         len(manifest["train_families"]) >= 6 and
         len(set(manifest["train_families"])) == len(manifest["train_families"]) and
         not (set(manifest["train_families"]) &
              (set(VALIDATION_SEEDS) | set(SCORE_SEEDS) | set(HELD_V25))),
         "Bank/provenance/reserve contract differs")
    read_closed_v34(closure_path)
    with np.load(bank_path, allow_pickle=False) as bank:
        obs = torch.from_numpy(bank["actor_observation"].copy()).float()
        labels = torch.from_numpy(bank["raw_action"].copy()).float()
        families = bank["seed"].tolist()
    need(obs.shape == (len(families), 500, 73) and
         labels.shape == (len(families), 500, 21) and
         families == manifest["train_families"], "Bank matrix differs")
    source = torch.load(V25_CHECKPOINT, map_location="cpu", weights_only=True)
    actor = source["actor_state_dict"]
    widths = [(512, 73), (256, 512), (128, 256), (21, 128)]
    need(all(tuple(actor[f"mlp.{2*i}.weight"].shape) == shape
             for i, shape in enumerate(widths)) and
         tuple(actor["obs_normalizer._mean"].shape) == (1, 73) and
         tuple(actor["obs_normalizer._std"].shape) == (1, 73),
         "Saved V25 actor architecture/normalizer differs")
    mean = actor["obs_normalizer._mean"].float()
    std = actor["obs_normalizer._std"].float()
    x = (obs - mean) / (std + 0.01)
    need(torch.isfinite(x).all().item(), "Normalized actor input nonfinite")

    def model_from_saved() -> nn.Sequential:
        model = nn.Sequential(nn.Linear(73, 512), nn.ELU(), nn.Linear(512, 256),
                              nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 21))
        own = model.state_dict()
        for key in own:
            own[key] = actor[f"mlp.{key}"].clone()
        model.load_state_dict(own)
        return model

    # Every family contributes equally; half of each batch comes from the first
    # second, so settled tail frames cannot drown the vulnerable rise stage.
    def run_fit(keep: list[int], epochs: int, fit_seed: int) -> nn.Sequential:
        torch.manual_seed(fit_seed)
        model = model_from_saved()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        generator = torch.Generator().manual_seed(fit_seed)
        for _ in range(epochs):
            model.train()
            for _ in range(16):
                epi = torch.tensor(keep)[torch.randint(len(keep), (512,), generator=generator)]
                early = torch.randint(0, 50, (256,), generator=generator)
                tail = torch.randint(50, 500, (256,), generator=generator)
                steps = torch.cat((early, tail))
                pred = model(x[epi, steps])
                loss = (pred - labels[epi, steps]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return model.eval()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    # Epoch selection is whole-family held-out *within* the development bank.
    # It is never a claim about the untouched learner validation/score reserve.
    candidates = [n for n in (2, 4, 8, 12, 20, 30) if n <= max_epochs]
    if max_epochs not in candidates:
        candidates.append(max_epochs)
    candidate_scores = []
    for epochs in sorted(candidates):
        fold_scores = []
        for hold in range(len(families)):
            model = run_fit([i for i in range(len(families)) if i != hold], epochs,
                            seed + hold)
            with torch.inference_mode():
                early = (model(x[hold, :50]) - labels[hold, :50]).square().mean().item()
                tail = (model(x[hold, 50:]) - labels[hold, 50:]).square().mean().item()
            fold_scores.append({"family": families[hold], "early_mse": early,
                                "tail_mse": tail, "balanced_mse": (early + tail) / 2})
        candidate_scores.append({"epochs": epochs, "folds": fold_scores,
                                 "mean_balanced_mse": sum(f["balanced_mse"] for f in fold_scores)
                                                      / len(fold_scores)})
    chosen = min(candidate_scores, key=lambda row: (row["mean_balanced_mse"], row["epochs"]))
    model = run_fit(list(range(len(families))), chosen["epochs"], seed)
    trained = {key: value.clone() for key, value in actor.items()}
    for key, value in model.state_dict().items():
        trained[f"mlp.{key}"] = value.detach().clone()
    # Load with load_cfg={"actor": True, "critic": True}; no stale PPO optimizer
    # or env state is exported. PPO continuation must initialize a new optimizer.
    with checkpoint.open("xb") as output:
        torch.save({"actor_state_dict": trained,
                    "critic_state_dict": source["critic_state_dict"],
                    "iter": 0, "infos": {}}, output)
    report = {"schema": "robot-learning-lab.cove50-seated-bc.v35.cpu-fit",
              "claim_limit": "Supervised on-route fit only; zero native learner replays or GPU steps",
              "preparer_source_sha256": manifest["preparer_source_sha256"],
              "fitter_source_sha256": sha(Path(__file__)),
              "cpu_runtime": cpu_runtime(),
              "bank_manifest_sha256": sha(manifest_path), "checkpoint_sha256": sha(checkpoint),
              "initial_actor_checkpoint_sha256": sha(V25_CHECKPOINT),
              "architecture": "73-512-256-128-21 ELU, saved V25 normalizer fixed",
              "normalizer_rule": "(raw observation - saved mean)/(saved std + 0.01)",
              "input_clip": None, "adam_lr": 3e-4, "batches_per_epoch": 16,
              "batch_size": 512, "early_fraction": 0.5,
              "leave_whole_family_out": candidate_scores,
              "chosen_epochs": chosen["epochs"],
              "fresh_validation_families_still_unopened": list(VALIDATION_SEEDS),
              "separate_score_families_still_unopened": list(SCORE_SEEDS)}
    with fit_report_path.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"chosen_epochs": chosen["epochs"],
                      "loo_balanced_mse": chosen["mean_balanced_mse"],
                      "checkpoint_sha256": sha(checkpoint), "native_replays": 0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect-v26")
    build = sub.add_parser("build-bank")
    build.add_argument("--v34-closure", type=Path, required=True)
    training = sub.add_parser("fit")
    training.add_argument("--bank", type=Path, default=V35 / "bank.npz")
    training.add_argument("--manifest", type=Path, default=V35 / "bank-manifest.json")
    training.add_argument("--max-epochs", type=int, default=12)
    training.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()
    if args.command == "inspect-v26":
        inspect_v26()
    elif args.command == "build-bank":
        build_bank(args.v34_closure)
    else:
        fit(args.bank, args.manifest, args.max_epochs, args.seed)


if __name__ == "__main__":
    main()
