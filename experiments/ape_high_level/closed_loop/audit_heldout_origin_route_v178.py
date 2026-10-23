"""V178: frozen V152 actor route selection in held-out origin images."""

from __future__ import annotations

import json
from pathlib import Path

import audit_origin_route_v152 as route
from run import ROOT, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v178"
V176 = ROOT / "artifacts/ape-closed-loop-cove50-v176/heldout-feasible-scene-audit.json"
V177 = ROOT / "artifacts/ape-closed-loop-cove50-v177/heldout-origin-audit.json"
FRAMES = ROOT / "artifacts/ape-closed-loop-cove50-v177"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V176.read_text())["rows"]
    entries = json.loads(V177.read_text())["rows"]
    rows = []
    for scene in scenes:
        seed = scene["seed"]
        positions = scene["positions_xy_m"]
        variant = {"variant": f"seed{seed}-heldout-feasible",
                   "cream_cheese_xy_m": positions["CreamCheese"],
                   "parmesan_xy_m": positions["Parmesan"]}
        entry = next(row for row in entries if row["seed"] == seed and
                     not row["absent"] and row["light_multiplier"] == 1.)
        row = route.screen(seed, variant, entry, frame_root=FRAMES,
                           light_multiplier=1.)
        rows.append(row)
        print(seed, row["status"], row.get("selected_kind"),
              "min", None if row.get("evaluator_selected_path") is None
              else row["evaluator_selected_path"]["min_full_scene_m"],
              flush=True)
    result = {"scope": "Frozen V152 self-masked RGB-D + 4/8/11cm actor 9cm point route selection from V177 held-out origin sweeps, scenes89,90,93,94; evaluator path score afterward only. Native light; no controller tuning.",
              "source_sha256": sha(SOURCE),
              "v152_import_source_sha256": sha(route.SOURCE),
              "v176_scene_audit_sha256": sha(V176),
              "v177_origin_audit_sha256": sha(V177), "rows": rows}
    (OUT / "heldout-origin-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
