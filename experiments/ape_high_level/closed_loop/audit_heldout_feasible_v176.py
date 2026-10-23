"""V176: evaluator-only predeclared moved-prop held-out feasibility screen."""

from __future__ import annotations

import json
from pathlib import Path

from audit_heldout_rooms_v175 import audit_scene
from run import ROOT, Room, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v176"
V175 = ROOT / "artifacts/ape-closed-loop-cove50-v175/heldout-room-validity.json"


def props(seed: int) -> dict[str, tuple[float, float]]:
    return ({"CreamCheese": (.40, .60), "Parmesan": (.64, .72)}
            if seed % 2 else
            {"CreamCheese": (.40, .60), "Parmesan": (.76, -.72)})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        rows = [audit_scene(room, seed, prop_overrides=props(seed), output_dir=OUT)
                for seed in (89, 90, 93, 94)]
    finally:
        room.close()
    for row in rows:
        row["valid_feasible"] = (row["object_scene_gap_m"] >= .03 and
                                  row["fixed_yaw_route_found"] and
                                  row["route_min_body_clearance_m"] is not None and
                                  row["route_min_body_clearance_m"] >= .065)
    result = {"scope": "Predeclared new seeds89,90,93,94; parity-fixed moved distractors from development scene generator, native seeded targets/furniture/light, evaluator-only 3cm object-scene and 6.5cm fixed-yaw full-body route feasibility. No actor outcomes.",
              "source_sha256": sha(SOURCE), "v175_geometry_audit_sha256": sha(V175),
              "v175_import_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/audit_heldout_rooms_v175.py"),
              "rows": rows}
    (OUT / "heldout-feasible-scene-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "object gap", row["object_scene_gap_m"],
              "route", row["fixed_yaw_route_found"],
              "min", row["route_min_body_clearance_m"],
              "valid feasible", row["valid_feasible"], flush=True)


if __name__ == "__main__":
    main()
