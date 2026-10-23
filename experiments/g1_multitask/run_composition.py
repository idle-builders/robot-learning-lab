"""Repeat the one-skill i4h/Cosmos step until stop, blocked, or a small bound."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("demo")
    p.add_argument("frame", type=int)
    p.add_argument("goal")
    p.add_argument("output", type=Path)
    p.add_argument("--i4h", type=Path, required=True)
    p.add_argument("--cosmos", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--tray-model", type=Path, required=True)
    p.add_argument("--cart-model", type=Path, required=True)
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--first-native-reset", action="store_true")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source, demo, frame = args.source, args.demo, args.frame
    results = []
    for turn in range(args.max_turns):
        folder = args.output / f"turn-{turn + 1}"
        command = [sys.executable, str(Path(__file__).with_name("advance_skill.py")),
                   str(source), demo, str(frame), args.goal, str(folder),
                   "--i4h", str(args.i4h), "--cosmos", str(args.cosmos), "--adapter", str(args.adapter),
                   "--tray-model", str(args.tray_model), "--cart-model", str(args.cart_model),
                   "--max-steps", "1600" if turn == 0 else "1100"]
        if turn == 0 and args.first_native_reset:
            command.append("--native-reset")
            if args.seed is not None:
                command += ["--seed", str(args.seed)]
        subprocess.run(command, check=True)
        result = json.loads((folder / "summary.json").read_text())
        results.append(result)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        if result["skill"] == "stop" or "blocked" in result:
            break
        source = folder / "rollout.hdf5"
        with h5py.File(source) as file:
            demo = next(name for name, episode in file["data"].items() if "states" in episode)
            frame = len(file[f"data/{demo}/actions"]) - 1
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
