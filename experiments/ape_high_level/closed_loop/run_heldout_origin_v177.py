"""V177: frozen V151 owner-crop origin sweeps in predeclared held-out rooms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_color_grounded_entry_v151 as entry
from run import ROOT, Vision, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v177"
V176 = ROOT / "artifacts/ape-closed-loop-cove50-v176/heldout-feasible-scene-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V176.read_text())["rows"]
    if [row["seed"] for row in scenes] != [89, 90, 93, 94]:
        raise AssertionError("Unexpected held-out scene list")
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    old_out = entry.OUT
    entry.OUT = OUT
    rows = []
    try:
        for scene in scenes:
            seed = scene["seed"]
            positions = scene["positions_xy_m"]
            variant = {"variant": f"seed{seed}-heldout-feasible",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            for absent in (False, True):
                row = entry.episode(seed, variant, absent, 1., vision, exemplar, owner)
                rows.append(row)
                print(seed, "absent", absent, "correct", row["correct_acceptances"],
                      "wrong", row["wrong_acceptances"],
                      "look alarm", row["look_range_alarm"], flush=True)
    finally:
        entry.OUT = old_out
    result = {"scope": "Unchanged V151 owner-confirmed exemplar + DINO0.35/hue0.70 origin lower RGB-D five-angle guarded sweep, native seed lighting, predeclared V176 feasible scenes89,90,93,94, paired present/absent. No base motion and no actor access to evaluator truth.",
              "source_sha256": sha(SOURCE),
              "v151_import_source_sha256": sha(entry.SOURCE),
              "v176_scene_audit_sha256": sha(V176),
              "owner_exemplar_sha256": sha(V149), "owner_color_audit_sha256": sha(V150),
              "rows": rows}
    (OUT / "heldout-origin-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
