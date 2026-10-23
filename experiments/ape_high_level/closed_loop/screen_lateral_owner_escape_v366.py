"""V366: can a side-only pre-invitation move replace the blind retreat?

Evaluator-only geometry from V359's revealed hard person case. A side move
could be observed by existing flank sensors, but this screen does not claim
that the actor has seen its full body/joint swept volume.
"""

from __future__ import annotations

import json
from pathlib import Path

import screen_owner_reposition_v359 as screen
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v366"
SIDE_MOVES_XY_M = ((0., -.04), (0., -.06), (0., -.08), (0., -.10),
                   (0., .06), (0., .10))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [screen.run_case(move, variant, reach, out=OUT)
            for move in SIDE_MOVES_XY_M for variant in screen.VARIANTS
            for reach in screen.REACHES]
    summary = []
    for move in SIDE_MOVES_XY_M:
        subset = [r for r in rows if r["retreat_xy_m"] == move]
        summary.append({
            "side_move_xy_m": move,
            "all_four_meet_5cm_margin": all(
                r["evaluator_meets_5cm_margin"] for r in subset),
            "worst_person_gap_m": min(
                r["evaluator_min_person_gap_m"] for r in subset),
            "worst_prop_gap_m": min(
                r["evaluator_min_prop_gap_m"] for r in subset),
            "worst_scene_gap_m": min(
                r["evaluator_min_scene_gap_m"] for r in subset),
        })
    report = {
        "scope": "Evaluator-only 20ms selected Cove-50 screen from the V353 guarded owner-help pose. Freeze -0.4rad shoulder and compare pure side moves 4/6/8/10cm negative and 6/10cm positive before a far or closer ignored near offer, for base and revealed left-shifted/thicker person. This is revealed development geometry, not actor-observed free space or a native lateral skill. One 8cm surrogate move is the current search limit; 10cm would require replanning and another step.",
        "source_sha256": sha(SOURCE),
        "screen_source_sha256": sha(Path(screen.__file__)),
        "summary": summary, "rows": rows}
    (OUT / "lateral-escape-screen.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for item in summary:
        print(item, flush=True)


if __name__ == "__main__":
    main()
