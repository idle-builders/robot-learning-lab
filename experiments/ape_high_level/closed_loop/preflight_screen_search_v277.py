"""V277: reserve a new mirrored screen split after V276 dropout diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

from direct_structural_rooms_v270 import make_room, preflight, selected_visual
from preflight_screen_search_v275 import lateral_witness
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v277"
SEED = 134
MODES = ("present", "absent", "identical_lookalike")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in MODES:
        room = make_room(SEED, "screen", mode)
        try:
            rows.append({"seed": SEED, "mode": mode,
                         "scene_preflight": preflight(room),
                         "bilateral_lateral_witness": lateral_witness(room)})
        finally:
            room.close()
    report = {"scope": "V277 predeclare fresh mirrored screen seed134 and present/ordinary absent/identical lookalike before V278 tuned actor exposure. Same exact Direct Frame scene generator. Evaluator-only bilateral 0-16cm side-step body clearance; actor gets no target/occluder truth or witness direction. V276 revealed seeds129/132 are development only. Frozen V278 adjustment from V276 is a 2-of-3 healthy/clear lateral-beam rule under 2.5% independent dropout; other task/identity/view thresholds unchanged.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "fresh-mirror-preflight.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], row["mode"], row["bilateral_lateral_witness"],
              flush=True)


if __name__ == "__main__":
    main()
