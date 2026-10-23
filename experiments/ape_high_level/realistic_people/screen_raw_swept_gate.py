"""Saved-packet screen of an independent immediate low-scan command veto."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import random

from raw_swept_gate import assess_swept_low_hits


EPISODES = (143, 185, 155, 177, 194)


def source(root: Path, version: int, filename: str) -> Path:
    paths = glob.glob(str(root / f"*-v{version}" / "evidence" / filename))
    if len(paths) != 1:
        raise FileNotFoundError((version, filename, paths))
    return Path(paths[0])


def replay(root: Path, version: int, seed: int | None = None,
           noise_fraction: float = 0., dropout_fraction: float = 0.) -> dict:
    ranges = json.loads(source(root, version, "range-fan-trace.json").read_text())["frames"]
    actions = json.loads(source(root, version, "online-action-trace.json").read_text())["actions"]
    rng = random.Random(seed)
    rows = []
    for frame, (scan, action) in enumerate(zip(ranges, actions)):
        command = action["command_for_next_frame"]
        step = command["forward_step_m"]
        if step <= 0:
            continue
        beams = scan["planes"][0]["beams"]
        low = [beam["distance_m"] if beam["distance_m"] is not None
               else float("inf") for beam in beams]
        if seed is not None:
            low = [None if rng.random() < dropout_fraction else
                   (max(.001, value + rng.gauss(0., noise_fraction * value))
                    if value != float("inf") else value)
                   for value in low]
        result = assess_swept_low_hits(
            low, [beam["angle_deg"] for beam in beams], step_m=step,
            step_heading_rel_deg=(command["translation_heading_deg"] -
                                  scan["body_heading_deg_commanded"]))
        rows.append({"frame": frame, **result})
    return {"version": version, "rows": rows,
            "veto_frames": [row["frame"] for row in rows if row["veto"]]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("artifacts/ape-realistic-people"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    baseline = [replay(args.root, v) for v in EPISODES]
    faults = []
    for noise, dropout in ((.05, .05), (.10, .10)):
        for version in EPISODES:
            trials = [replay(args.root, version, seed=seed,
                             noise_fraction=noise, dropout_fraction=dropout)["veto_frames"]
                      for seed in range(20)]
            faults.append({"version": version, "noise": noise,
                           "dropout": dropout, "veto_frames_by_seed": trials})
    sources = {str(version): {name: {"path": str(path),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for name in ("range-fan-trace.json", "online-action-trace.json")
                              for path in [source(args.root, version, name)]}
               for version in EPISODES}
    result = {"scope": "unchanged historical actions; occupied-hit veto counterfactual",
              "clearance_m": .24, "sources": sources,
              "baseline": baseline, "faults": faults}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"baseline": {x["version"]: x["veto_frames"] for x in baseline},
                      "faults": [{"version": x["version"], "noise": x["noise"],
                                  "veto_12": sum(12 in frames for frames in x["veto_frames_by_seed"]),
                                  "any_veto": sum(bool(frames) for frames in x["veto_frames_by_seed"])}
                                 for x in faults]}, indent=2))


if __name__ == "__main__":
    main()
