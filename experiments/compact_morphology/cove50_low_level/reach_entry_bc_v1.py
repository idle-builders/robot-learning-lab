"""Audit saved entry-locked reach visits and fit one command-conditioned arm pilot.

This only trains the four-joint arm residual. The native posture, authored arm
path, observed entry goal and route command remain separate controller inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_arm_residual import ArmResidualActor


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = ROOT / "artifacts/ape-low-level-training-20261014/reach-entry-bc-v1"
MATRICES = (BASE / "v152/native-paired-routes/matrix.json",
            BASE / "v158/native-development-contact/matrix.json")
ACTORS = (BASE / "v57/actor.pt", BASE / "v10/actor.pt",
          BASE / "v15/actor-locked.pt")
MODES = ("v16_locked", "v16_specialist_locked", "v21_scoped_y_locked")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor(path: Path) -> ArmResidualActor:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = ArmResidualActor(saved["input_dim"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def entries() -> list[dict]:
    chosen = []
    for matrix_path in MATRICES:
        matrix = json.loads(matrix_path.read_text())
        # V152 permanently has one invalid original; its 47 retained valid
        # pairs are the source set. V158 completed all 18 originals.
        if matrix_path == MATRICES[0]:
            if matrix["case_count"] != 47 or matrix["planned_count"] != 48:
                raise ValueError(f"Unexpected V152 source boundary: {matrix_path}")
        elif not matrix["complete"] or matrix["case_count"] != 18:
            raise ValueError(f"Incomplete source: {matrix_path}")
        for case in matrix["cases"]:
            if case.get("numeric_rgb_entry_faults") or case.get("status") not in (
                    None, "valid_matched_native_pair"):
                continue
            first = "baseline" if "baseline_report" in case else "automatic"
            choice = first if case[f"{first}_margin_1p2_s_met"] else "prescribed"
            if not case[f"{choice}_margin_1p2_s_met"]:
                raise ValueError(f"No successful donor in family {case['seed']}")
            report_path = ROOT / case[f"{choice}_report"]
            trace_path = ROOT / case[f"{choice}_trace"]
            if sha(report_path) != case[f"{choice}_report_sha256"] or sha(trace_path) != case[f"{choice}_trace_sha256"]:
                raise ValueError(f"Source hash mismatch: {case['seed']}")
            chosen.append({"seed": case["seed"], "choice": choice,
                           "report": report_path, "trace": trace_path})
    if len(chosen) != 65 or len({row["seed"] for row in chosen}) != 65:
        raise ValueError(f"Expected 65 independent donor families, got {len(chosen)}")
    return chosen


def feature_rows(report: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [row for row in report["timeseries"] if row["phase"] == "reach"]
    if len(rows) != 450 or report["selection_mode"] not in MODES or not report["image_goal_lock_after_entry"]:
        raise ValueError("Wrong reach episode/route/goal contract")
    error = np.asarray([r["controller_target_error_xy_m"] for r in rows], dtype=np.float32) / .01
    integral = np.asarray([r["learned_error_integral"] for r in rows], dtype=np.float32)
    correction = np.asarray([r["learned_arm_residual_raw"] for r in rows], dtype=np.float32)
    previous = np.vstack((np.zeros((1, 4), dtype=np.float32), correction[:-1] / .1))
    progress = np.clip(np.arange(450, dtype=np.float32) * .02 / 4.0, 0, 1)
    blend = progress * progress * (3 - 2 * progress)
    x = np.column_stack((np.repeat(error[:1], 450, axis=0), error, integral,
                         previous, blend)).astype(np.float32)
    command = np.zeros((450, len(MODES)), dtype=np.float32)
    command[:, MODES.index(report["selection_mode"])] = 1
    return np.column_stack((x, command)), correction, blend


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(14651)
    np.random.seed(14651)
    actor_map = {sha(path): actor(path) for path in ACTORS}
    source = actor(ACTORS[0])
    episodes = []
    for item in entries():
        report = json.loads(item["report"].read_text())
        checkpoint = report["selected_reach_checkpoint_sha256"]
        if checkpoint not in actor_map:
            raise ValueError(f"Unknown source actor {checkpoint} for {item['seed']}")
        with np.load(item["trace"]) as trace:
            if len(trace["action"]) != 875 or len(trace["qpos"]) != 875:
                raise ValueError(f"Incomplete native trace: {item['seed']}")
            if hashlib.sha256(trace["action"][0].astype(np.float32).tobytes()).hexdigest() != report["first_action_sha256"]:
                raise ValueError(f"First executed action mismatch: {item['seed']}")
        x, correction, blend = feature_rows(report)
        with torch.no_grad():
            original = actor_map[checkpoint](torch.from_numpy(x[:, :11])).numpy()
        residual_gap = np.abs(original * (.1 * blend[:, None]) - correction)
        if not np.all(np.isfinite(x)) or float(residual_gap.max()) > 5e-4:
            raise ValueError(f"Actor/controller parity failed: {item['seed']} {residual_gap.max()}")
        # The saved correction was applied before the physical step. Near zero
        # blend, rounding dominates; retain the original actor's output there.
        target = original.copy()
        reliable = blend >= .1
        target[reliable] = correction[reliable] / (.1 * blend[reliable, None])
        target = np.clip(target, -1, 1).astype(np.float32)
        episodes.append({**item, "mode": report["selection_mode"], "x": x,
                         "y": target, "source_output": original,
                         "max_residual_gap": float(residual_gap.max()),
                         "p99_residual_gap": float(np.quantile(residual_gap, .99))})
    by_mode = {mode: [row for row in episodes if row["mode"] == mode] for mode in MODES}
    # Whole-family holdout from each route stratum, fixed before fitting.
    valid_seeds = {row["seed"] for mode in MODES for row in sorted(
        by_mode[mode], key=lambda r: hashlib.sha256(str(r["seed"]).encode()).digest())[:max(2, round(len(by_mode[mode]) * .2))]}
    train = [row for row in episodes if row["seed"] not in valid_seeds]
    valid = [row for row in episodes if row["seed"] in valid_seeds]
    def pack(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(np.concatenate([r["x"] for r in rows])),
                torch.from_numpy(np.concatenate([r["y"] for r in rows])))
    tx, ty = pack(train)
    vx, vy = pack(valid)
    model = ArmResidualActor(14)
    state = model.state_dict()
    previous = source.state_dict()
    for key in state:
        if key == "net.0.weight":
            state[key][:, :11] = previous[key]
            state[key][:, 11:] = 0
        else:
            state[key] = previous[key].clone()
    model.load_state_dict(state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, 61):
        model.train()
        for _ in range(20):
            index = torch.randint(len(tx), (512,))
            loss = torch.mean((model(tx[index]) - ty[index]) ** 2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(torch.mean((model(vx) - vy) ** 2).sqrt())
        history.append({"epoch": epoch, "held_family_rmse": round(value, 6)})
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch >= 12 and epoch - best_epoch >= 10:
            break
    model.load_state_dict(best_state)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT / "actor.pt"
    torch.save({"state_dict": best_state, "input_dim": 14, "residual_bound": .1,
                "route_commands": MODES, "source_actor_sha256": sha(ACTORS[0]),
                "method": "entry_locked_command_conditioned_arm_residual_bc_v1"}, checkpoint_path)
    with torch.no_grad():
        held = model(vx)
        original_held = source(vx[:, :11])
        baseline_rmse = float(torch.mean((original_held - vy) ** 2).sqrt())
        predicted_rmse = float(torch.mean((held - vy) ** 2).sqrt())
        held_by_mode = {}
        for mode in MODES:
            group = [row for row in valid if row["mode"] == mode]
            xx, yy = pack(group)
            routed = torch.from_numpy(np.concatenate([row["source_output"] for row in group]))
            held_by_mode[mode] = {
                "families": len(group),
                "candidate_rmse": float(torch.mean((model(xx) - yy) ** 2).sqrt()),
                "original_routed_actor_rmse": float(torch.mean((routed - yy) ** 2).sqrt()),
            }
    report = {"schema": "robot-learning-lab.reach-entry-bc-v1.v1",
              "qualification": "Offline command-conditioned arm imitation only; no fresh native result",
              "source_matrices": {str(p.relative_to(ROOT)): sha(p) for p in MATRICES},
              "source_actor_sha256": {str(p.relative_to(ROOT)): sha(p) for p in ACTORS},
              "episode_count": len(episodes), "train_families": len(train),
              "held_development_families": len(valid), "train_rows": len(tx),
              "held_rows": len(vx), "mode_counts": dict(Counter(r["mode"] for r in episodes)),
              "donor_choice_counts": dict(Counter(r["choice"] for r in episodes)),
              "max_actor_correction_gap": max(r["max_residual_gap"] for r in episodes),
              "max_p99_actor_correction_gap": max(r["p99_residual_gap"] for r in episodes),
              "held_baseline_v57_rmse": baseline_rmse,
              "held_candidate_rmse": predicted_rmse,
              "held_by_mode": held_by_mode,
              "best_epoch": best_epoch, "optimizer_updates": len(history) * 20,
              "checkpoint_sha256": sha(checkpoint_path), "validation_seeds": sorted(valid_seeds),
              "sources": [{"seed": r["seed"], "mode": r["mode"], "choice": r["choice"],
                           "split": "held_development" if r["seed"] in valid_seeds else "train",
                           "report": str(r["report"].relative_to(ROOT)),
                           "report_sha256": sha(r["report"]),
                           "trace": str(r["trace"].relative_to(ROOT)),
                           "trace_sha256": sha(r["trace"])} for r in episodes],
              "history": history}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("episode_count", "train_families",
          "held_development_families", "mode_counts", "max_actor_correction_gap",
          "held_baseline_v57_rmse", "held_candidate_rmse", "best_epoch",
          "optimizer_updates", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
