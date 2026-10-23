"""V263: frozen owner-taught origin sweeps in V262's varied rooms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_color_grounded_entry_v151 as entry
from audit_varied_dynamic_rooms_v262 import VariedFurnishedRoom
from run import ROOT, Vision, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v263"
V262 = ROOT / "artifacts/ape-closed-loop-cove50-v262/varied-dynamic-room-preflight.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


def main() -> None:
    scenes = json.loads(V262.read_text())["rows"]
    eligible = [row for row in scenes if row["label"] == "route_valid_dynamic"]
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old = (entry.OUT, entry.MovedPropRoom)
    entry.OUT = OUT
    entry.MovedPropRoom = VariedFurnishedRoom
    rows = []
    try:
        for scene in eligible:
            seed = scene["seed"]
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            for absent in (False, True):
                variant = {"variant": f"seed{seed}-varied-dynamic-{'absent' if absent else 'present'}",
                           "cream_cheese_xy_m": positions["CreamCheese"],
                           "parmesan_xy_m": (scene["absent_challenge"]["parmesan_xy_m"]
                                              if absent else positions["Parmesan"])}
                result = entry.episode(seed, variant, absent, 1., vision,
                                       exemplar, owner_hist)
                rows.append(result)
                print(seed, "absent", absent, "correct",
                      result["correct_acceptances"], "wrong",
                      result["wrong_acceptances"], flush=True)
    finally:
        entry.OUT, entry.MovedPropRoom = old
    report = {"scope": "V263 first actor exposure in V262 evaluator-preflighted route-valid varied furnished rooms only. Frozen V151 owner-confirmed exemplar, DINO>=.35 and owner hue>=.70 over five lower-head RGB-D views; paired true can and near-memory target-absent Parmesan, same native seeded light and V262 wall/furniture variations. 10Hz virtual range guards during head looks. No base motion, physical optics, red same-color lookalike or new threshold tuning.",
              "source_sha256": sha(SOURCE), "v262_sha256": sha(V262),
              "eligible_seeds": [row["seed"] for row in eligible],
              "rows": rows}
    (OUT / "varied-room-origin-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
