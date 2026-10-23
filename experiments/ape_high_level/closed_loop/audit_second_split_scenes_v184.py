"""V184: geometry-only second split after seed-93 development repair."""

from __future__ import annotations

import json
from pathlib import Path

from audit_heldout_rooms_v175 import audit_scene
from audit_heldout_feasible_v176 import props
from run import ROOT, Room, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v184"
V183 = ROOT / "artifacts/ape-closed-loop-cove50-v183/origin-to-target-audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        rows = [audit_scene(room, seed, prop_overrides=props(seed), output_dir=OUT)
                for seed in (95, 96, 97, 98)]
    finally:
        room.close()
    for row in rows:
        row["valid_feasible"] = (row["object_scene_gap_m"] >= .03 and
                                  row["fixed_yaw_route_found"] and
                                  row["route_min_body_clearance_m"] is not None and
                                  row["route_min_body_clearance_m"] >= .065)
    result = {"scope": "Second independent split seeds95-98, V176 parity-fixed moved distractors, native seeded target/furniture/light, evaluator-only 3cm object-scene and 6.5cm fixed-yaw full-body 21-sample route feasibility before actor exposure.",
              "source_sha256": sha(SOURCE), "v183_development_audit_sha256": sha(V183),
              "v175_import_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/audit_heldout_rooms_v175.py"),
              "v176_props_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/audit_heldout_feasible_v176.py"),
              "rows": rows}
    (OUT / "second-split-scene-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "object gap", row["object_scene_gap_m"],
              "route", row["fixed_yaw_route_found"],
              "min", row["route_min_body_clearance_m"],
              "valid feasible", row["valid_feasible"], flush=True)


if __name__ == "__main__":
    main()
