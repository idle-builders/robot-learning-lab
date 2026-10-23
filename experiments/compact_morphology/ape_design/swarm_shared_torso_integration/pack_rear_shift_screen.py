"""Check whether rearward 4S-pack relocation frees a +16 mm hip-yaw shift.

This is a static envelope screen only. No hatch, wiring, retention or moving
leg is modeled, so a clear row is not a selected system architecture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as b

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_hip_yaw_rethink.hip_yaw_rethink import assembly_solids, shell
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.check_common_volume import pair_report

OUT = ROOT / "artifacts/shared-torso-integration/pack-rear-shift.json"


def main():
    _, skin, _ = shell(500, 2)
    original = assembly_solids(500, "remote", 200, 225)
    hip = [(name, b.Pos(0, 0, 16) * shape)
           for name, shape in original["fixed"] + original["moving"]
           if "battery" not in name and "service" not in name]
    service = [(name, shape) for name, shape in original["fixed"] if "service" in name]
    pack_source = next(shape for name, shape in original["fixed"] if "battery" in name)
    rows = []
    for rear_mm in (0, 10, 14, 16, 18, 20):
        pack = [("4S candidate pack", b.Pos(0, rear_mm, 0) * pack_source)]
        against_hip = pair_report(pack, hip)
        against_skin = pair_report(pack, list(skin.items()))
        against_service = pair_report(pack, service)
        rows.append({"pack_rear_shift_mm": rear_mm,
                     "hip": against_hip,
                     "skin": against_skin,
                     "service": against_service})
        print(rear_mm, len(against_hip["intersections"]),
              len(against_skin["intersections"]),
              len(against_service["intersections"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"height_mm": 500, "wall_mm": 2,
                               "hip_shift_up_mm": 16,
                               "scope": "static volumes only; no battery hatch or moving leg",
                               "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
