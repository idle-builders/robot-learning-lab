"""Small paired variation screen of the retained shared BC actor.

This is a development screen, not independent qualification or a retrain.
It reuses the sealed V1 native scorer and sources; no V1 score is overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import score_shared_actor as v1


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHOD = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn-breadth-v2/method.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, default=DEFAULT_METHOD)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    if method["schema"] != "robot-learning-lab.shared-forward-turn.breadth-v2.method.v1":
        raise ValueError("wrong method")
    for item in method["sources"].values():
        path = ROOT / item["path"]
        if not path.is_file() or v1.sha(path) != item["sha256"]:
            raise ValueError(f"source changed: {path}")
    train_method = json.loads((ROOT / method["sources"]["train_method"]["path"]).read_text())
    for item in train_method["sources"].values():
        path = ROOT / item["path"]
        if not path.is_file() or v1.sha(path) != item["sha256"]:
            raise ValueError(f"V1 training source changed: {path}")
    v1.RUN = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn/native"
    v1.OUT = args.out.resolve()
    if v1.OUT.exists():
        raise RuntimeError("original output already exists")
    v1.OUT.mkdir(parents=True)
    rows = []
    for case in method["cases"]:
        pair = []
        for mode in ("source", "bc"):
            row = v1.score_case(case, mode, v1.OUT, method, train_method)
            pair.append(row)
            rows.append(row)
            (v1.OUT / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(case["name"], mode, row["pass"], row["failure_reasons"], flush=True)
        if pair[0]["initial_state_sha256"] != pair[1]["initial_state_sha256"]:
            raise RuntimeError(f"unmatched start: {case['name']}")
    result = {"schema": "robot-learning-lab.shared-forward-turn.breadth-v2.result.v1",
              "claim_limit": "Fresh paired development variation screen of experimental Direct Frame motion; not broad robustness, motor safety or physical transfer",
              "method_sha256": v1.sha(args.method), "rows": rows,
              "pairs": len(method["cases"]), "visits": len(rows),
              "bc_passes": sum(row["pass"] for row in rows if row["mode"] == "bc"),
              "source_passes": sum(row["pass"] for row in rows if row["mode"] == "source"),
              "optimizer_updates": 0}
    (v1.OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pairs": result["pairs"], "bc_passes": result["bc_passes"],
                      "source_passes": result["source_passes"]}), flush=True)


if __name__ == "__main__":
    main()
