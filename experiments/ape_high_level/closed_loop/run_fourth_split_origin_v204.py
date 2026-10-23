"""V204: frozen owner origin in V203 present and near-memory absent rooms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_color_grounded_entry_v151 as entry
from run import ROOT, Vision, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v204"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


def main() -> None:
    scenes = json.loads(V203.read_text())["rows"]
    assert [row["seed"] for row in scenes] == list(range(107, 115))
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    old_out = entry.OUT
    entry.OUT = OUT
    rows = []
    try:
        for scene in scenes:
            if scene["label"].startswith("invalid"):
                continue
            seed = scene["seed"]
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            present = {"variant": f"seed{seed}-{scene['label']}-present",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            absent = {"variant": f"seed{seed}-{scene['label']}-nearby-absent",
                      "cream_cheese_xy_m": positions["CreamCheese"],
                      "parmesan_xy_m": scene["absent_challenge"]["parmesan_xy_m"]}
            for missing, variant in ((False, present), (True, absent)):
                row = entry.episode(seed, variant, missing, 1., vision, exemplar, owner)
                rows.append({"geometry_label": scene["label"], **row})
                print(seed, scene["label"], "absent", missing,
                      "correct", row["correct_acceptances"],
                      "wrong", row["wrong_acceptances"], flush=True)
    finally:
        entry.OUT = old_out
    result = {"scope": "Unchanged V151 owner-confirmed exemplar DINO/hue lower RGB-D five-angle 10Hz-range-guarded origin sweep, native seed light, all V203 paired-geometry-valid seeds107-114. Present parity props; matched absent Parmesan moved to true present target XY. No base motion/evaluator truth as actor input.",
              "source_sha256": sha(SOURCE),
              "v151_source_sha256": sha(entry.SOURCE),
              "v203_sha256": sha(V203),
              "owner_exemplar_sha256": sha(V149),
              "owner_color_audit_sha256": sha(V150),
              "rows": rows}
    (OUT / "fourth-split-origin-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
