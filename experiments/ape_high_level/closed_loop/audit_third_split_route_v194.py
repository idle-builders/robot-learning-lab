"""V194: unchanged V152 actor route screen on frozen V193 origin frames."""

from __future__ import annotations

import json
from pathlib import Path

import audit_origin_route_v152 as route
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v194"
V192 = ROOT / "artifacts/ape-closed-loop-cove50-v192/third-split-geometry-audit.json"
V193 = ROOT / "artifacts/ape-closed-loop-cove50-v193/third-split-origin-audit.json"
FRAMES = ROOT / "artifacts/ape-closed-loop-cove50-v193"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V192.read_text())["rows"]
    entries = json.loads(V193.read_text())["rows"]
    rows = []
    for scene in scenes:
        if scene["label"] == "invalid_object_separation":
            continue
        seed = scene["seed"]
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        variant = {"variant": f"seed{seed}-{scene['label']}",
                   "cream_cheese_xy_m": positions["CreamCheese"],
                   "parmesan_xy_m": positions["Parmesan"]}
        entry = next(row for row in entries if row["seed"] == seed and
                     not row["absent"] and row["light_multiplier"] == 1.)
        row = route.screen(seed, variant, entry, frame_root=FRAMES,
                           light_multiplier=1.)
        rows.append({"geometry_label": scene["label"], **row})
        print(seed, scene["label"], row["status"],
              row.get("selected_kind"), flush=True)
    result = {"scope": "Unchanged V152 self-masked RGB-D plus virtual4/8/11cm actor 9cm point route selection from V193 owner-grounded origin sweeps, all V192 geometry-valid seeds; evaluator selected path score afterward only.",
              "source_sha256": sha(SOURCE),
              "v152_source_sha256": sha(route.SOURCE),
              "v192_sha256": sha(V192), "v193_sha256": sha(V193),
              "rows": rows}
    (OUT / "third-split-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
