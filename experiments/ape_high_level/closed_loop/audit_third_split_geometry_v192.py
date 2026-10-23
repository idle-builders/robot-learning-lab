"""V192: evaluator-only 6.5/12 cm route strata before actor exposure."""

from __future__ import annotations

import json
from pathlib import Path

from audit_heldout_rooms_v175 import audit_scene
from audit_heldout_feasible_v176 import props
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v192"
SEEDS = tuple(range(99, 107))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    room = Room()
    try:
        for seed in SEEDS:
            audits = {str(margin): audit_scene(
                room, seed, prop_overrides=props(seed), output_dir=OUT,
                model_margin_m=margin) for margin in (.065, .12)}
            narrow, generous = audits["0.065"], audits["0.12"]
            assert narrow["object_scene_gap_m"] == generous["object_scene_gap_m"]
            valid = narrow["object_scene_gap_m"] >= .03
            if not valid:
                label = "invalid_object_separation"
            elif generous["fixed_yaw_route_found"]:
                label = "generous_kinematic"
            elif narrow["fixed_yaw_route_found"]:
                label = "tight_kinematic"
            else:
                label = "blocked_fixed_yaw"
            rows.append({"seed": seed, "label": label, "audits": audits})
            print(seed, label, "object gap", narrow["object_scene_gap_m"],
                  "12cm route", generous["fixed_yaw_route_found"], flush=True)
    finally:
        room.close()
    result = {"scope": "Predeclared seeds99-106, V176 parity-fixed moved props, V175 evaluator-only all-solid object separation and fixed-yaw 4cm/21-sample witness path, 6.5cm and12cm strata. No actor sensor/control input.",
              "source_sha256": sha(SOURCE),
              "v175_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/audit_heldout_rooms_v175.py"),
              "v176_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/audit_heldout_feasible_v176.py"),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "seeds": list(SEEDS), "rows": rows}
    (OUT / "third-split-geometry-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
