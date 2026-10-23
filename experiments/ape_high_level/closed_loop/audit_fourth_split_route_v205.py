"""V205: unchanged V152 actor route screen on frozen V204 present frames."""

from __future__ import annotations

import json
from pathlib import Path

import audit_origin_route_v152 as route
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v205"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
FRAMES = ROOT / "artifacts/ape-closed-loop-cove50-v204"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    rows = []
    for scene in scenes:
        if scene["label"].startswith("invalid"):
            continue
        seed = scene["seed"]
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        variant = {"variant": f"seed{seed}-{scene['label']}-present",
                   "cream_cheese_xy_m": positions["CreamCheese"],
                   "parmesan_xy_m": positions["Parmesan"]}
        entry = next(row for row in entries if row["seed"] == seed and
                     not row["absent"] and row["light_multiplier"] == 1.)
        row = route.screen(seed, variant, entry, frame_root=FRAMES,
                           light_multiplier=1.)
        rows.append({"geometry_label": scene["label"], **row})
        print(seed, scene["label"], row["status"],
              row.get("selected_kind"), flush=True)
    result = {"scope": "Unchanged V152 self-masked RGB-D plus virtual4/8/11cm actor 9cm point route selection from V204 owner-grounded present origin sweeps, all V203 paired-geometry-valid seeds; evaluator selected path score afterward only.",
              "source_sha256": sha(SOURCE),
              "v152_source_sha256": sha(route.SOURCE),
              "v203_sha256": sha(V203), "v204_sha256": sha(V204),
              "rows": rows}
    (OUT / "fourth-split-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
