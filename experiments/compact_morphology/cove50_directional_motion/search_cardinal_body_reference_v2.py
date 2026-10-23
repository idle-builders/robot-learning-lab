"""One bounded local sensitivity screen around the V1 side-release boundary.

Development feasibility only: one correlated CPU reset, no learned policy or
native CUDA demonstration. Do not use this as an independent skill score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_cardinal_body_reference import BODY, make_env, sha, visit


CASES = [
    {"id": f"{direction}-boundary-{i}", "direction": direction,
     "swing": direction, "lean": lean, "knee_lift": knee, "travel": travel}
    for direction in ("left", "right")
    for i, (lean, knee, travel) in enumerate((
        (.10, .30, .20), (.11, .30, .20), (.12, .30, .20),
        (.10, .35, .25), (.11, .35, .25), (.12, .35, .25)), 1)
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    env = make_env(seed=49401)
    results = []
    try:
        for case in CASES:
            row = visit(env, case, seed=49401)
            results.append(row)
            print(case["id"], row["steps"],
                  round(row["signed_progress_m"] * 1000, 1),
                  row["one_sole_frames"], row["local_physical_screen_pass"],
                  flush=True)
    finally:
        env.close()
    report = {
        "schema": "robot-learning-lab.cardinal-body-reference-boundary-v2.v1",
        "claim_limit": "Correlated one-seed local CPU scripted-target screen; no learned policy, accepted imitation example or CUDA skill score",
        "body_sha256": sha(BODY), "source_sha256": sha(Path(__file__)),
        "v1_source_sha256": sha(Path(__file__).with_name("search_cardinal_body_reference.py")),
        "seed": 49401, "evaluated_cases": len(results),
        "local_physical_screen_passes": sum(r["local_physical_screen_pass"] for r in results),
        "gpu_hours": 0, "optimizer_updates": 0, "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
