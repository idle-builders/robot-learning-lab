"""V189: evaluator-only 9/9.5 cm feasibility of V187 fixed scenes."""

from __future__ import annotations

import json
from pathlib import Path

from audit_heldout_rooms_v175 import audit_scene
from audit_heldout_feasible_v176 import props
from run import ROOT, Room, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v189"
V184 = ROOT / "artifacts/ape-closed-loop-cove50-v184/second-split-scene-audit.json"
V187 = ROOT / "artifacts/ape-closed-loop-cove50-v187/origin-to-target-audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    rows = []
    try:
        for seed in (96, 97, 98):
            for margin in (.09, .095):
                row = audit_scene(room, seed, prop_overrides=props(seed),
                                  output_dir=OUT, model_margin_m=margin)
                rows.append(row)
                print(seed, margin, "route", row["fixed_yaw_route_found"],
                      "min", row["route_min_body_clearance_m"],
                      "searched", row["sampled_poses"], flush=True)
    finally:
        room.close()
    result = {"scope": "Evaluator-only fixed-neutral-yaw second-split scenes96-98, same 4cm-grid 21-sample full-body origin route search as V184 but 9.0 and9.5cm margins to match actor route conservatism. Diagnostic, no actor input and no retroactive scene reclassification.",
              "source_sha256": sha(SOURCE), "v184_sha256": sha(V184),
              "v187_sha256": sha(V187),
              "v175_import_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/audit_heldout_rooms_v175.py"),
              "rows": rows}
    (OUT / "margin-matched-feasibility-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
