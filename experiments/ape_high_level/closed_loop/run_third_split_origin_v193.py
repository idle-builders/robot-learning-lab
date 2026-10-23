"""V193: frozen V151 owner-grounded origin sweeps on V192 rooms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_color_grounded_entry_v151 as entry
from run import ROOT, Vision, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v193"
V192 = ROOT / "artifacts/ape-closed-loop-cove50-v192/third-split-geometry-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


def main() -> None:
    scenes = json.loads(V192.read_text())["rows"]
    assert [row["seed"] for row in scenes] == list(range(99, 107))
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    old_out = entry.OUT
    entry.OUT = OUT
    rows = []
    try:
        for scene in scenes:
            if scene["label"] == "invalid_object_separation":
                continue
            seed = scene["seed"]
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            variant = {"variant": f"seed{seed}-{scene['label']}",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            for absent in (False, True):
                row = entry.episode(seed, variant, absent, 1., vision, exemplar, owner)
                rows.append(row)
                print(seed, scene["label"], "absent", absent,
                      "correct", row["correct_acceptances"],
                      "wrong", row["wrong_acceptances"], flush=True)
    finally:
        entry.OUT = old_out
    result = {"scope": "Unchanged V151 owner-confirmed exemplar DINO/hue lower RGB-D five-angle 10Hz-range-guarded origin sweep, native seed light, all V192 geometry-valid seeds99-106. Present/absent paired, no base motion, no actor evaluator truth.",
              "source_sha256": sha(SOURCE),
              "v151_source_sha256": sha(entry.SOURCE),
              "v192_sha256": sha(V192),
              "owner_exemplar_sha256": sha(V149),
              "owner_color_audit_sha256": sha(V150),
              "rows": rows}
    (OUT / "third-split-origin-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
