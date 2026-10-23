"""Unscored CUDA reward, command and full-state warm-start gate for V3."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/compact_morphology/mjlab_ape/src"))
sys.path.insert(0, str(ROOT))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from ape_walk import task as ape  # noqa: E402
from experiments.compact_morphology.cove50_policy_transfer import train_velocity  # noqa: E402

ART = ROOT / "artifacts/ape-low-level-training-20261015/cardinal-v3"
OUT = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/cardinal-v3/sentinel")
BASE = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v2/candidate/candidate-final.pt"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
TRAINER = ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_cfg():
    captured = {}
    old_launch, old_argv = train_velocity.launch_training, sys.argv
    try:
        train_velocity.launch_training = lambda task, cfg: captured.update(task=task, cfg=cfg)
        sys.argv = [str(TRAINER), "--body-xml", str(BODY), "--profile", "cardinal_move",
                    "--worlds", "1024", "--updates", "24", "--run-name", "cardinal-v3-sentinel",
                    "--resume-run", "cardinal-v3-sentinel-source", "--resume-checkpoint", "model_883.pt",
                    "--ankle-target-margin-rad", "0.02",
                    "--directional-progress-weight", "2.0", "--walking-hip-roll-pose-std", "0.2"]
        train_velocity.main()
    finally:
        train_velocity.launch_training, sys.argv = old_launch, old_argv
    return captured["cfg"]


def main() -> None:
    if not Path("/workspace").is_mount() or OUT.exists():
        raise RuntimeError("Fresh persistent sentinel output required")
    method = json.loads((ART / "method.json").read_text())
    if (sha(BASE) != method["source"]["v2_checkpoint"]["sha256"]
            or sha(TRAINER) != method["source"]["trainer"]["sha256"]
            or sha(BODY) != method["source"]["body"]["sha256"]):
        raise RuntimeError("Source differs from frozen cardinal method")
    OUT.mkdir(parents=True)
    cfg = capture_cfg()
    cfg.env.seed = cfg.agent.seed = method["cuda_entry"]["seed"]
    assert cfg.env.scene.num_envs == 1024 and cfg.env.episode_length_s == 8.0
    if (cfg.env.rewards["signed_directional_progress"].weight != 2.0
            or cfg.env.rewards["pose"].params["std_walking"][r".*hip_roll.*"] != 0.2):
        raise RuntimeError("Resolved directional reward or hip-roll prior differs")
    with (OUT / "native-rollout.log").open("x") as log:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = log
            env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0"),
                                    clip_actions=cfg.agent.clip_actions)
            try:
                runner = ape.VelocityOnPolicyRunner(env, asdict(cfg.agent), device="cuda:0")
                runner.load(str(BASE), strict=True, map_location="cuda:0")
                assert runner.current_learning_iteration == 883
                obs = env.get_observations()
                command = env.unwrapped.command_manager.get_term("twist")
                initial = command.command[:, :3].clone()
                counts = {
                    "backward": int(((initial[:, 0] < -.1) & (initial[:, 1] == 0)).sum()),
                    "left": int(((initial[:, 1] > .1) & (initial[:, 0] == 0)).sum()),
                    "right": int(((initial[:, 1] < -.1) & (initial[:, 0] == 0)).sum()),
                    "forward": int(((initial[:, 0] > .1) & (initial[:, 1] == 0)).sum()),
                }
                same_command_all_steps = torch.ones(env.num_envs, device="cuda:0", dtype=torch.bool)
                expected_command = initial.clone()
                pending_reset = torch.zeros(env.num_envs, device="cuda:0", dtype=torch.bool)
                zero_command_world_steps = 0
                reset_count = 0
                early_reset_count = 0
                torch.cuda.synchronize()
                start = time.monotonic()
                for step in range(400):
                    live_command = command.command[:, :3]
                    expected_command[pending_reset] = live_command[pending_reset]
                    live = live_command[:, :2].abs().amax(dim=1) > 1e-8
                    same = (live_command - expected_command).abs().amax(dim=1) < 1e-7
                    same_command_all_steps &= live & same
                    zero_command_world_steps += int((~live).sum())
                    with torch.inference_mode():
                        action = runner.alg.actor(obs)
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("Nonfinite cardinal actor action")
                    obs, reward, dones, _ = env.step(action)
                    if not bool(torch.isfinite(reward).all()) or not bool(torch.isfinite(obs["actor"]).all()):
                        raise RuntimeError("Nonfinite cardinal native rollout")
                    reset_count += int(dones.sum())
                    if step < 399:
                        early_reset_count += int(dones.sum())
                    pending_reset = dones.bool().reshape(-1).clone()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                memory = torch.cuda.max_memory_allocated() / 2**30
            finally:
                env.close()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    if (sum(counts.values()) != 1024
            or int(same_command_all_steps.sum()) != 1024
            or zero_command_world_steps):
        raise RuntimeError("Cardinal command timing or bucket coverage failed on CUDA")
    # Exercise the exact body/evaluator/graphics path before optimizer work.
    video = OUT / "video-path.mp4"
    native_report = OUT / "video-path.json"
    native_trace = OUT / "video-path.npz"
    with (OUT / "video-path.log").open("x") as log:
        proc = subprocess.run([sys.executable, str(EVALUATOR), str(BASE),
            "--task", "Mjlab-Velocity-Flat-Ape-Elbows",
            "--body-xml", str(BODY), "--physics-model", "frame50",
            "--initial-root-z", ".231", "--yaw-only-head", "--arms-down",
            "--worlds", "1", "--seed", str(method["cuda_entry"]["seed"]), "--velocity-x", "-.25",
            "--velocity-y", "0", "--angular-z", "0",
            "--motor-rail-v", "12", "--command-delay-ms", "0",
            "--ankle-target-margin-rad", ".02", "--physics-dt", ".002",
            "--duration-s", "4.04",
            "--out", str(native_report), "--diagnostic-trace", str(native_trace),
            "--video", str(video)], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, timeout=400, check=False)
    if proc.returncode:
        raise RuntimeError(f"Cardinal exact-body video sentinel exited {proc.returncode}")
    import imageio.v2 as imageio
    import numpy as np
    decoded = 0
    minimum_std = float("inf")
    with imageio.get_reader(str(video), format="ffmpeg") as reader:
        for frame in reader:
            pixel_std = float(np.std(frame))
            if frame.shape != (480, 640, 3) or pixel_std < 1.0:
                raise RuntimeError(f"Blank or malformed cardinal video frame {decoded}")
            minimum_std = min(minimum_std, pixel_std)
            decoded += 1
    opened = json.loads(native_report.read_text())
    if (opened["seed"] != method["cuda_entry"]["seed"] or opened["body_sha256"] != sha(BODY)
            or not native_trace.is_file() or decoded < 50):
        raise RuntimeError("Cardinal exact-body video or native source mismatch")
    source_run = ROOT / "logs/rsl_rl/ape_cove50_velocity/cardinal-v3-sentinel-source"
    source_run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(BASE, source_run / "model_883.pt")
    before = {p.name for p in source_run.parent.iterdir() if p.is_dir()}
    with (OUT / "one-update.log").open("x") as log:
        proc = subprocess.run([sys.executable, str(TRAINER), "--body-xml", str(BODY),
            "--profile", "cardinal_move", "--worlds", "1024", "--updates", "1",
            "--save-interval", "1", "--run-name", "cardinal-v3-disposable",
            "--resume-run", "cardinal-v3-sentinel-source", "--resume-checkpoint", "model_883.pt",
            "--ankle-target-margin-rad", ".02", "--directional-progress-weight", "2.0",
            "--walking-hip-roll-pose-std", ".2"], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, timeout=1200, check=False)
    if proc.returncode:
        raise RuntimeError(f"Disposable cardinal update exited {proc.returncode}")
    new = [p for p in source_run.parent.iterdir() if p.is_dir() and p.name not in before]
    if len(new) != 1 or not (new[0] / "model_883.pt").is_file():
        raise RuntimeError("Disposable optimizer checkpoint absent or ambiguous")
    candidate = new[0] / "model_883.pt"
    state = torch.load(candidate, map_location="cpu", weights_only=False)
    if state.get("iter") != 883 or sha(candidate) == sha(BASE):
        raise RuntimeError("Disposable optimizer state did not advance")
    report = {
        "schema": "robot-learning-lab.cardinal-v3.cuda-sentinel.v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "method_sha256": sha(ART / "method.json"),
        "source_checkpoint_sha256": sha(BASE), "body_sha256": sha(BODY),
        "worlds": 1024, "native_steps_per_world": 400,
        "initial_bucket_counts": counts,
        "worlds_with_stable_nonzero_command_per_episode_all_400_steps": int(same_command_all_steps.sum()),
        "zero_command_world_steps": zero_command_world_steps,
        "early_reset_count": early_reset_count,
        "reset_count": reset_count, "rollout_elapsed_s": round(elapsed, 3),
        "native_world_steps_per_second": round(1024 * 400 / elapsed, 1),
        "peak_cuda_memory_gib": round(memory, 3),
        "decoded_video_frames": decoded,
        "minimum_video_frame_pixel_std": round(minimum_std, 3),
        "video_path_report_sha256": sha(native_report),
        "video_path_trace_sha256": sha(native_trace),
        "video_path_mp4_sha256": sha(video),
        "disposable_optimizer_updates": 1,
        "disposable_checkpoint_sha256": sha(candidate),
        "candidate_optimizer_updates": 0,
        "files": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
        "claim_limit": "Unscored CUDA command-time and disposable optimizer sentinel; not candidate training or a physical skill score.",
    }
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
