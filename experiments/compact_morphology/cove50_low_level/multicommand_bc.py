"""Small shared proprioceptive BC start for bounded Direct Frame commands."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
TASKS = ("squat", "both_arms")


class Actor(nn.Module):
    def __init__(self, observation_width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_width + len(TASKS), 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 21),
        )

    def forward(self, observation: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((observation, task_id), dim=-1))


class HeadsActor(nn.Module):
    def __init__(self, observation_width: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_width + len(TASKS), 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.heads = nn.ModuleList(nn.Linear(128, 21) for _ in TASKS)

    def forward(self, observation: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        features = self.trunk(torch.cat((observation, task_id), dim=-1))
        return sum(head(features) * task_id[:, index:index + 1]
                   for index, head in enumerate(self.heads))


def load_student(path: Path, device: str, target_names: list[str],
                 action_scale: torch.Tensor | float, action_offset: torch.Tensor | float,
                 task: str):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["body_sha256"] != BODY_SHA or checkpoint["tasks"] != TASKS:
        raise ValueError("student body or command contract differs")
    if (checkpoint["target_names"] != target_names or task not in TASKS
            or task not in checkpoint.get("trained_tasks", TASKS)):
        raise ValueError("student joint ordering or task differs")
    actor_class = HeadsActor if checkpoint.get("separate_heads", False) else Actor
    model = actor_class(len(checkpoint["observation_mean"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = checkpoint["observation_mean"].to(device)
    std = checkpoint["observation_std"].to(device)
    target_mean = checkpoint["target_mean"].to(device)
    target_std = checkpoint["target_std"].to(device)
    ident = torch.zeros(1, len(TASKS), device=device)
    ident[0, TASKS.index(task)] = 1.0
    scale = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
    offset = torch.as_tensor(action_offset, dtype=torch.float32, device=device)
    safe_scale = torch.where(scale.abs() > 1e-8, scale, 1.0)

    def policy(observations):
        with torch.inference_mode():
            actor = observations["actor"]
            if actor.shape[1] != len(mean):
                raise ValueError("student observation width differs")
            if checkpoint.get("mask_action_history", False):
                actor = actor.clone()
                actor[:, 48:69] = 0.0
            target = model((actor - mean) / std, ident.expand(len(actor), -1))
            target = target * target_std + target_mean
            raw = (target - offset) / safe_scale
            return torch.where(scale.abs() > 1e-8, raw, 0.0)

    return policy


def load_episode(item: dict) -> dict:
    trace = Path(item["trace"])
    report = Path(item["report"])
    check = Path(item["check"])
    checked = json.loads(check.read_text())
    if checked["trace_sha256"] != hashlib.sha256(trace.read_bytes()).hexdigest():
        raise ValueError(f"trace changed after check: {trace}")
    if checked["report_sha256"] != hashlib.sha256(report.read_bytes()).hexdigest():
        raise ValueError(f"report changed after check: {report}")
    if checked["status"] != "trace_clock_and_shapes_valid; physical_teacher_gate_separate":
        raise ValueError(f"invalid trace check: {check}")
    result = json.loads(report.read_text())
    if result["body_sha256"] != BODY_SHA or result["device"] != "cpu":
        raise ValueError("this first fit expects source-labelled CPU demonstrations")
    if item["task"] == "squat":
        first, middle, last = (result.get(name) for name in
                               ("initial_hold", "middle_hold", "final_hold"))
        passes = (result["duration_s"] >= 9.99 and not result["ended_early"]
                  and all((first, middle, last))
                  and first["mean_pelvis_z_m"] - middle["mean_pelvis_z_m"] >= 0.04
                  and abs(first["mean_pelvis_z_m"] - last["mean_pelvis_z_m"]) <= 0.015
                  and all(part["both_feet_grounded_fraction"] >= 0.95
                          and part["hand_contact_frames"] == 0
                          and part["body_contact_fraction"] == 0
                          and part["saturated_leg_fraction"] <= 0.1
                          for part in (first, middle, last)))
    elif item["task"] == "both_arms":
        first, raised, last = (result.get(name) for name in
                               ("before", "raised", "lowered"))
        passes = (result["duration_s"] >= 9.99 and all((first, raised, last))
                  and min(raised["hand_height_m"][i] - first["hand_height_m"][i]
                          for i in range(2)) >= 0.15
                  and max(abs(last["hand_height_m"][i] - first["hand_height_m"][i])
                          for i in range(2)) <= 0.04
                  and result["peak_hand_force_n"] < 1.0
                  and all(part["shoulder_saturation_fraction"] <= 0.1
                          for part in (first, raised, last)))
    else:
        raise ValueError(f"unknown task {item['task']}")
    if not passes:
        raise ValueError(f"teacher missed its basic whole-episode gate: {report}")
    with np.load(trace, allow_pickle=False) as data:
        observation = data["actor_observation"].astype(np.float32)
        scale = data["action_scale"].astype(np.float32)
        offset = data["action_offset"].astype(np.float32)
        target = data["unsmoothed_action"].astype(np.float32) * scale + offset
        names = data["target_names"].tolist()
    if observation.shape[1] != 72 or target.shape[1] != 21:
        raise ValueError("expected common 72-observation, 21-target contract")
    return {"observation": observation, "target": target,
            "active": (abs(scale) > 1e-8).astype(np.float32), "target_names": names,
            "task": item["task"], "split": item["split"], "seed": result["seed"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--mask-action-history", action="store_true",
                        help="Ablate teacher-action history from the 72D actor observation")
    parser.add_argument("--separate-heads", action="store_true",
                        help="Shared proprioceptive trunk with one physical-target head per command")
    parser.add_argument("--single-task", choices=TASKS,
                        help="Fit one independent actor with its own observation normalizer")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    episodes = [load_episode(item) for item in manifest["episodes"]
                if args.single_task is None or item["task"] == args.single_task]
    if args.mask_action_history:
        for item in episodes:
            item["observation"] = item["observation"].copy()
            item["observation"][:, 48:69] = 0.0
    trained_tasks = (args.single_task,) if args.single_task else TASKS
    if {item["task"] for item in episodes} != set(trained_tasks):
        raise ValueError("requested command examples are missing")
    names = episodes[0]["target_names"]
    if any(item["target_names"] != names for item in episodes):
        raise ValueError("joint ordering differs between teachers")
    for task in trained_tasks:
        if not any(item["task"] == task and item["split"] == "train" for item in episodes):
            raise ValueError(f"missing {task} train family")
        if not any(item["task"] == task and item["split"] == "validation" for item in episodes):
            raise ValueError(f"missing {task} validation family")
    train = [item for item in episodes if item["split"] == "train"]
    held = [item for item in episodes if item["split"] == "validation"]
    if {item["seed"] for item in train} & {item["seed"] for item in held}:
        raise ValueError("train/validation seed overlap")
    obs = np.concatenate([item["observation"] for item in train])
    targets = np.concatenate([item["target"] for item in train])
    mean = torch.from_numpy(obs.mean(axis=0))
    std = torch.from_numpy(obs.std(axis=0).clip(0.05))
    target_mean = torch.from_numpy(targets.mean(axis=0))
    target_std = torch.from_numpy(targets.std(axis=0).clip(0.05))
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    model = (HeadsActor if args.separate_heads else Actor)(obs.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def batch(items):
        x = torch.from_numpy(np.concatenate([i["observation"] for i in items]))
        y = torch.from_numpy(np.concatenate([i["target"] for i in items]))
        mask = torch.from_numpy(np.concatenate([
            np.broadcast_to(i["active"], i["target"].shape) for i in items
        ]).copy())
        ident = torch.cat([
            torch.eye(len(TASKS))[TASKS.index(i["task"])].repeat(len(i["target"]), 1)
            for i in items
        ])
        return (x - mean) / std, (y - target_mean) / target_std, mask, ident

    x, y, mask, ident = batch(train)
    vx, vy, vmask, vident = batch(held)
    best = float("inf")
    best_step = 0
    best_model = None
    generator = torch.Generator().manual_seed(20261015)
    for step in range(args.updates):
        sample = torch.randint(len(x), (256,), generator=generator)
        prediction = model(x[sample], ident[sample])
        weight = 0.1 + 0.9 * mask[sample]
        loss = ((prediction - y[sample]).square() * weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 20 == 0:
            with torch.no_grad():
                value = (((model(vx, vident) - vy).square() * (0.1 + 0.9 * vmask)).mean()).item()
            if value < best:
                best, best_step = value, step + 1
                best_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert best_model is not None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_model, "observation_mean": mean,
                "observation_std": std, "target_mean": target_mean,
                "target_std": target_std, "tasks": TASKS,
                "target_names": names, "body_sha256": BODY_SHA,
                "mask_action_history": args.mask_action_history,
                "separate_heads": args.separate_heads,
                "trained_tasks": trained_tasks}, args.checkpoint)
    summary = {"schema": "robot-learning-lab.multicommand-bc-pilot.v1",
               "claim_limit": "CPU on-route imitation fit only; fresh closed-loop physical score separate",
               "body_sha256": BODY_SHA,
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "train_episodes": len(train), "validation_episodes": len(held),
               "train_frames": int(len(x)), "validation_frames": int(len(vx)),
               "optimizer_updates": args.updates, "best_update": best_step,
               "mask_action_history": args.mask_action_history,
               "separate_heads": args.separate_heads,
               "trained_tasks": trained_tasks,
               "held_weighted_normalized_mse": best,
               "gpu_hours": 0, "listed_gpu_cost_usd": 0}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
