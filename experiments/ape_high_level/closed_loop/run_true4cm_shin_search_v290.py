"""V290: live four-shin owner search against the genuinely 4 cm floor object."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_direct_structural_entry_v271 import V149, V150
from audit_true_low_obstacle_v289 import set_full_height
from preflight_lateral_clutter_v282 import make_room
from run import ROOT, Vision, sha
from run_lateral_clutter_shin_search_v284 import run_case


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v290"
V289 = ROOT / "artifacts/ape-closed-loop-cove50-v289/true-low-obstacle-audit.json"
V284_SOURCE = ROOT / "experiments/ape_high_level/closed_loop/run_lateral_clutter_shin_search_v284.py"
SEED = 138
MODES = ("present", "absent", "identical_lookalike")
FULL_HEIGHT_M = .04


def room_factory(seed: int, mode: str):
    room = make_room(seed, mode)
    set_full_height(room, FULL_HEIGHT_M)
    return room


def main() -> None:
    prior = json.loads(V289.read_text())
    frozen = prior["valid_start_scores"]["4cm"]
    if frozen["shin_four"]["false_safe"] != 0:
        raise ValueError("The 4 cm static shin screen changed")
    if not next(row for row in prior["preflight"]
                if row["height_m"] == FULL_HEIGHT_M)["starts"][0]["valid_start_5cm"]:
        raise ValueError("Start is not clear")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for mode in MODES:
        row = run_case(SEED, mode, vision, exemplar, owner_hist,
                       room_factory=room_factory, output_root=OUT,
                       sensor_case_prefix="v290-true4cm")
        rows.append(row)
        (OUT / "true4cm-shin-search-audit.json").write_text(
            json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
        print(mode, row["status"], "clearance",
              row["evaluator_min_model_clearance_m"], flush=True)
    report = {"scope": "V290 same V284 owner-facing visual search and articulated four-shin point-only route gate in selected Direct Frame screen seed138, but the floor box is actually 4cm tall (MuJoCo half-height2cm). Frozen V289 static valid-start screen showed no unsafe passes but one safe false-block at the second first-side step. Present/ordinary absent/identical lookalike, 8cm kinematic steps with 0.88 slip, estimated pose and frozen DINO/hue owner exemplar. Evaluator clearance and instance truth only after actor decisions. This is a local behavior comparison, not full swept-free coverage, native walking, or a deployable package qualification.",
              "source_sha256": sha(SOURCE), "v284_source_sha256": sha(V284_SOURCE),
              "v289_sha256": sha(V289), "physical_crate_height_m": FULL_HEIGHT_M,
              "rows": rows}
    (OUT / "true4cm-shin-search-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
