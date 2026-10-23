"""V210: evaluator-only route headroom screen from the saved native stop tail."""

from __future__ import annotations

import json
from pathlib import Path

from audit_heldout_rooms_v175 import audit_scene
from audit_heldout_feasible_v176 import props
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v210"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
NATIVE = ROOT / "artifacts/ape-skill-contract-cove50-20261009/stop-summary-matrix.json"
BODY_CLEARANCE_M = .05


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    native = json.loads(NATIVE.read_text())["cases"]
    max_path = max(float(row["path_to_8s_m"]) for row in native)
    candidate = BODY_CLEARANCE_M + max_path
    assert .164 <= candidate <= .166
    margin = .165
    room = Room()
    rows = []
    try:
        for scene in scenes:
            if scene["label"] != "generous_kinematic":
                continue
            seed = scene["seed"]
            audit = audit_scene(room, seed, prop_overrides=props(seed),
                                output_dir=OUT, model_margin_m=margin)
            rows.append({"seed": seed,
                         "v203_12cm_witness": scene["audits"]["0.12"]["fixed_yaw_route_found"],
                         "tail_margin_witness": audit["fixed_yaw_route_found"],
                         "audit": audit})
            print(seed, "16.5cm route", audit["fixed_yaw_route_found"],
                  "expanded", audit["expanded_cells"], flush=True)
    finally:
        room.close()
    result = {"scope": "V210 evaluator-only fixed-posture fixed-yaw 4cm-grid 21-sample all-solid route screen in V203 generous seeds, margin16.5cm = rounded-up maximum observed native forward-policy path to8s(11.49cm)+5cm desired model clearance. No sensor sampling/cancel latency, gait-direction swept geometry, dynamic obstacle or high-level actor execution. This is a conservative stop-tail headroom diagnostic, not a calibrated stop envelope or physical safety qualification.",
              "source_sha256": sha(SOURCE), "v203_audit_sha256": sha(V203),
              "native_stop_summary_sha256": sha(NATIVE),
              "max_native_path_to_8s_m": max_path,
              "desired_remaining_model_clearance_m": BODY_CLEARANCE_M,
              "screen_margin_m": margin, "rows": rows}
    (OUT / "native-tail-margin-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
