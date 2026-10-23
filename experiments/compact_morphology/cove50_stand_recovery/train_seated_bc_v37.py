"""Refit the seated actor from ten source-checked whole scripted episodes.

V37 is an inexpensive learning update. It never evaluates its own checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-v37"
V35 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-v35"
V36 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-pilot-v36"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def main() -> None:
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(plan["status"] == "frozen_before_fit" and
            sha(Path(__file__)) == plan["trainer_sha256"] and
            plan["source_families"] == [35001, 35002, 35003, 35004, 35005, 35006,
                                        36101, 36102, 36103, 36104] and
            plan["epochs"] == 12 and plan["batches_per_epoch"] == 16 and
            plan["batch_size"] == 512 and plan["early_fraction"] == 0.5,
            "V37 method differs from freeze")
    model_path, report_path = ART / "model_bc_10family.pt", ART / "fit-report.json"
    require(not model_path.exists() and not report_path.exists(),
            "V37 original fit already exists")
    for name, path in (("v35_bank_sha256", V35 / "bank.npz"),
                       ("v35_bank_manifest_sha256", V35 / "bank-manifest.json"),
                       ("v35_fit_report_sha256", V35 / "fit-report.json"),
                       ("v36_plan_sha256", V36 / "plan.json"),
                       ("v36_source_checked_report_sha256", V36 / "source-checked-report.json"),
                       ("v25_checkpoint_sha256", ROOT / plan["v25_checkpoint"])):
        require(path.is_file() and sha(path) == plan[name], f"V37 source changed: {name}")
    manifest = json.loads((V35 / "bank-manifest.json").read_text())
    v36 = json.loads((V36 / "source-checked-report.json").read_text())
    v36_plan = json.loads((V36 / "plan.json").read_text())
    require(manifest["train_families"] == plan["source_families"][:6] and
            manifest["bank_sha256"] == plan["v35_bank_sha256"] and
            v36["complete_families"] == 4 and v36["scripted_full_gate"] == 4 and
            v36["plan_sha256"] == plan["v36_plan_sha256"] and
            [row["seed"] for row in v36["rows"]] == plan["source_families"][6:] and
            v36_plan["body_sha256"] == plan["body_sha256"] and
            v36_plan["baseline_checkpoint_sha256"] == plan["v25_checkpoint_sha256"],
            "V37 whole-family source or exact task differs")
    with np.load(V35 / "bank.npz", allow_pickle=False) as bank:
        observations = list(bank["actor_observation"].copy())
        labels = list(bank["raw_action"].copy())
        require(bank["seed"].tolist() == plan["source_families"][:6],
                "V35 bank seed order differs")
    provenance = []
    for row in v36["rows"]:
        seed = row["seed"]
        scored = row["scripted"]
        report_path_old = V36 / f"seed{seed}-scripted.json"
        trace_path = V36 / f"seed{seed}-scripted.npz"
        native = json.loads(report_path_old.read_text())
        require(scored["full_gate"] and not scored["failure_reasons"] and
                sha(report_path_old) == scored["source_report_sha256"] and
                sha(trace_path) == scored["native_trajectory_sha256"] and
                native["intervention_route"] == "scripted" and
                native["body_sha256"] == plan["body_sha256"] and
                native["evaluator_source_sha256"] == v36_plan["evaluator_sha256"] and
                native["duration_s"] == 10.0 and not native["ended_early"] and
                len(native["timeseries"]) == 500,
                f"V36 source/physical demonstration invalid: {seed}")
        with np.load(trace_path, allow_pickle=False) as trace:
            obs = trace["actor_observation"].copy()
            action = trace["raw_action"].copy()
            require(obs.shape == (500, 73) and action.shape == (500, 21) and
                    np.array_equal(action, trace["script_candidate_action"]) and
                    np.all(trace["selected_action_is_script"]) and
                    np.array_equal(trace["previous_raw_action"][1:], action[:-1]) and
                    np.isfinite(obs).all() and np.isfinite(action).all(),
                    f"V36 scripted pre-action alignment invalid: {seed}")
        observations.append(obs)
        labels.append(action)
        provenance.append({"seed": seed, "friction": native["initial_physics"]["sole_slide_friction"][0],
                           "source_report_sha256": sha(report_path_old),
                           "source_trace_sha256": sha(trace_path)})
    obs = torch.from_numpy(np.stack(observations).astype(np.float32))
    labels_tensor = torch.from_numpy(np.stack(labels).astype(np.float32))
    require(tuple(obs.shape) == (10, 500, 73) and
            tuple(labels_tensor.shape) == (10, 500, 21), "V37 bank shape differs")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    source = torch.load(ROOT / plan["v25_checkpoint"], map_location="cpu", weights_only=True)
    actor = source["actor_state_dict"]
    mean, std = actor["obs_normalizer._mean"].float(), actor["obs_normalizer._std"].float()
    x = (obs - mean) / (std + 0.01)
    require(torch.isfinite(x).all().item(), "V37 normalized observation nonfinite")
    model = nn.Sequential(nn.Linear(73, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                          nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 21))
    own = model.state_dict()
    for key in own:
        own[key] = actor[f"mlp.{key}"].clone()
    model.load_state_dict(own)
    torch.manual_seed(plan["training_seed"])
    generator = torch.Generator().manual_seed(plan["training_seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["adam_lr"])
    history = []
    for _ in range(plan["epochs"]):
        loss_sum = 0.0
        for _ in range(plan["batches_per_epoch"]):
            episodes = torch.randint(10, (512,), generator=generator)
            steps = torch.cat((torch.randint(0, 50, (256,), generator=generator),
                               torch.randint(50, 500, (256,), generator=generator)))
            loss = (model(x[episodes, steps]) - labels_tensor[episodes, steps]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        history.append(loss_sum / plan["batches_per_epoch"])
    model.eval()
    trained = {key: value.clone() for key, value in actor.items()}
    for key, value in model.state_dict().items():
        trained[f"mlp.{key}"] = value.detach().clone()
    with model_path.open("xb") as output:
        torch.save({"actor_state_dict": trained, "critic_state_dict": source["critic_state_dict"],
                    "iter": 0, "infos": {}}, output)
    with torch.inference_mode():
        predictions = model(x)
        family_mse = ((predictions - labels_tensor).square().mean(dim=(1, 2))).tolist()
    report = {"schema": "robot-learning-lab.seated-bc-v37.cpu-fit.v1",
              "claim_limit": "Ten-whole-family imitation fit only; zero V37 native rollout or robustness score",
              "plan_sha256": sha(plan_path), "trainer_sha256": sha(Path(__file__)),
              "checkpoint": str(model_path.relative_to(ROOT)),
              "checkpoint_sha256": sha(model_path),
              "source_families": plan["source_families"],
              "accepted_whole_episodes": 10, "training_frames": 5000,
              "cpu_optimizer_updates": plan["epochs"] * plan["batches_per_epoch"],
              "gpu_optimizer_updates": 0, "native_v37_visits": 0,
              "epoch_mean_training_mse": history, "per_family_training_mse": family_mse,
              "v36_added_demonstrations": provenance,
              "held_score_unopened": plan["held_score_unopened"]}
    with report_path.open("x") as output:
        output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("accepted_whole_episodes", "training_frames",
                       "cpu_optimizer_updates", "native_v37_visits", "checkpoint_sha256")}))


if __name__ == "__main__":
    main()
