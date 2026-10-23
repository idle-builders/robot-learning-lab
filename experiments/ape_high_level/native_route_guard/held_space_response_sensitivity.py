"""V32 held-space interaction sensitivity to simulated human response."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import held_space_request_loop as loop
import kitchen_native as kitchen
import kitchen_trial as trial


OUT = loop.OUT / "response-sensitivity"
DELAYS_S = (.25, .50, .75)
RETREAT_SPEEDS_MPS = (.40, .65, .90)


def main() -> None:
    scene = loop.late.previous.OUT / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    original_out = loop.OUT
    original_delay = loop.REACTION_S
    original_speed = loop.RETREAT_MPS
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            for delay in DELAYS_S:
                for speed in RETREAT_SPEEDS_MPS:
                    loop.OUT = OUT / f"delay{delay:.2f}-speed{speed:.2f}"
                    loop.REACTION_S = delay
                    loop.RETREAT_MPS = speed
                    episodes = []
                    for world in range(4):
                        for case in loop.CASES[:2]:
                            episodes.append(loop.episode(
                                model, saved, world, case,
                                "split_body_10hz", "complies", False))
                    row = {"reaction_s": delay, "retreat_mps": speed,
                           "episodes": episodes,
                           "requested": sum(x["request_s"] is not None
                                            for x in episodes),
                           "resumed": sum(x["resume_s"] is not None
                                          for x in episodes),
                           "below_5cm": sum(x["below_5cm"] for x in episodes),
                           "overlap": sum(x["overlap"] for x in episodes),
                           "worst_gap_m": min(x["min_body_person_gap_m"]
                                              for x in episodes)}
                    results.append(row)
                    print(delay, speed, row["resumed"], row["below_5cm"],
                          row["worst_gap_m"], flush=True)
    finally:
        loop.OUT = original_out
        loop.REACTION_S = original_delay
        loop.RETREAT_MPS = original_speed
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-held-space-response-sensitivity.v1",
              "scope": "Scripted human response sweep on the historical 50cm body. Same body-aware split low-range sensor and actor. Actor receives neither response parameters nor person ground truth. This tests only asking for space while the body replays its native held trace; it is not human safety qualification.",
              "source_sha256": trial.sha(Path(__file__)),
              "loop_source_sha256": trial.sha(Path(loop.__file__)),
              "scene_sha256": trial.sha(scene),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
