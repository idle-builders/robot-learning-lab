"""One reference G1 skill step: observe, select with Cosmos, execute with i4h."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py


def run(command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("demo")
    p.add_argument("frame", type=int)
    p.add_argument("goal")
    p.add_argument("output", type=Path)
    p.add_argument("--i4h", type=Path, required=True)
    p.add_argument("--cosmos", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--tray-model", type=Path)
    p.add_argument("--cart-model", type=Path, required=True)
    p.add_argument("--force-skill", choices=("tray_place", "cart_push"))
    p.add_argument("--max-steps", type=int, default=1100)
    p.add_argument("--native-reset", action="store_true")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent

    run([sys.executable, str(here / "observe_skill_scene.py"), str(args.source), args.demo,
         str(args.frame), args.goal, str(args.output / "observation")])
    if args.force_skill:
        skill = args.force_skill
        (args.output / "selection.json").write_text(json.dumps([{"raw_prediction": skill, "source": "fixed baseline"}], indent=2) + "\n")
    else:
        if not args.cosmos or not args.adapter or not args.tray_model:
            p.error("--cosmos, --adapter, and --tray-model are required without --force-skill")
        run([sys.executable, str(here / "cosmos_select_skill.py"), str(args.cosmos),
             str(args.output / "observation/case.json"), str(args.output / "selection.json"),
             "--adapter", str(args.adapter)], stdout=(args.output / "selector.log").open("w"))
        selection = json.loads((args.output / "selection.json").read_text())[0]
        skill = selection["raw_prediction"].strip()
    observed = json.loads((args.output / "observation/case.json").read_text())[0]
    choices = observed["available_skill_ids"]
    if skill not in choices:
        raise ValueError(f"Cosmos returned infeasible skill {skill!r}; available: {choices}")
    if skill == "stop":
        (args.output / "summary.json").write_text(json.dumps({"skill": skill, "source": str(args.source)}, indent=2) + "\n")
        print("selected stop; no motor run")
        return

    # Known entry conditions for these two reference skills; no floor-pickup skill exists.
    tray_on_cart = observed["tray_on_cart"]
    tray_on_shelf = observed["tray_xyz_m"][2] > -0.20 and observed["tray_xyz_m"][0] < -0.8
    if (skill == "cart_push" and not tray_on_cart) or (skill == "tray_place" and not tray_on_shelf):
        summary = {"skill": skill, "blocked": "selected skill entry condition is false",
                   "tray_on_cart": tray_on_cart, "tray_on_shelf": tray_on_shelf,
                   "source": str(args.source), "source_frame": args.frame}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
        return

    env_id, model = {
        "tray_place": ("locomanip_tray_pick_and_place", args.tray_model),
        "cart_push": ("locomanip_push_cart", args.cart_model),
    }[skill]
    handoff = args.output / "handoff.hdf5"
    if not args.native_reset:
        run([sys.executable, str(here / "extract_handoff_state.py"), str(args.source), args.demo,
             str(args.frame), str(handoff)])

    policy_cli = args.i4h / "workflows/agentic/policy/gr00t_n16/.venv/bin/i4h-agentic-gr00t-n16"
    arena_cli = args.i4h / "workflows/agentic/arena/.venv/bin/i4h-agentic-arena"
    policy_log = args.output / "policy.log"
    arena_log = args.output / "arena.log"
    policy_env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0"}
    with policy_log.open("w") as stream:
        policy = subprocess.Popen([str(policy_cli), "--env", env_id, "--model-path", str(model)],
                                  cwd=args.i4h, env=policy_env, stdout=stream, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 300
        while "policy IO ready" not in policy_log.read_text():
            if policy.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"policy did not become ready: {policy_log}")
            time.sleep(1)
        arena_env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "I4H_STRICT_NATIVE_SUCCESS": "1"}
        if args.native_reset:
            arena_env.pop("I4H_HANDOFF_HDF5", None)
        else:
            arena_env["I4H_HANDOFF_HDF5"] = str(handoff)
        command = [str(arena_cli), "--env", env_id, "--episodes", "1", "--max-timesteps", str(args.max_steps),
                   "--save-all-episodes", "--record-to", str(args.output / "rollout.hdf5"),
                   "--headless", "--num_envs", "1"]
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        with arena_log.open("w") as stream:
            run(command, cwd=args.i4h, env=arena_env,
                stdout=stream, stderr=subprocess.STDOUT)
    finally:
        policy.terminate()
        try:
            policy.wait(timeout=15)
        except subprocess.TimeoutExpired:
            policy.kill()
            policy.wait()

    with h5py.File(args.output / "rollout.hdf5") as file:
        episode = next(d for d in file["data"].values() if "states" in d)
        n = len(episode["actions"])
        cart = episode["states/articulation/cart/root_pose"][-1, :3].tolist()
        tray = episode["states/articulation/surgical_tray/root_pose"][-1, :3].tolist()
    log = arena_log.read_text()
    native_success = "Locomanip policy episode 1/1 attempt 1/1: completed" in log
    summary = {"skill": skill, "model": str(model), "source": str(args.source), "source_frame": args.frame,
               "native_reset": args.native_reset, "seed": args.seed,
               "native_success": native_success, "actions": n, "cart_xyz_m": cart, "tray_xyz_m": tray,
               "rollout": str(args.output / "rollout.hdf5")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
