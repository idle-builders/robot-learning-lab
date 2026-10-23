"""V333: hazard protection followed by the live scripted owner-help branch.

V317's present, absent and identical-imposter outcomes are exercised after
the V332 actor range guard in the same selected Direct Frame room. This is a
development test of task/hazard arbitration, not native physical interaction.
"""

from __future__ import annotations

import json
from pathlib import Path

from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
from run_late_corridor_intrusion_v323 import LateCorridorRoom
import run_live_hazard_guard_v332 as guard_module
from run_live_hazard_guard_v332 import LiveGuard, ParkedLateRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v333"
SEED = 170
CASES = tuple(case for case in owner_module.CASES if case[0] == SEED)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    original_room = owner_module.HeldoutMovingRoom
    original_run_case = owner_module.search.run_case
    original_out = owner_module.OUT
    original_cases = owner_module.CASES
    try:
        for condition, room_type in (("moving", LateCorridorRoom),
                                     ("parked", ParkedLateRoom)):
            guards = []

            def guarded_run_case(*args, **kwargs):
                owner_callback = kwargs["on_route_blocked"]
                seed, mode = args[:2]

                def guarded_callback(room, observe, branch, video):
                    guard = LiveGuard(room, branch, video, seed,
                                      f"seed{seed}-{mode}-{condition}")
                    guard.run()
                    response = owner_callback(room, observe, branch, video)
                    summary = guard.save(response["status"])
                    guards.append(summary)
                    response["hazard_guard"] = summary
                    return response

                kwargs["on_route_blocked"] = guarded_callback
                return original_run_case(*args, **kwargs)

            owner_module.HeldoutMovingRoom = room_type
            owner_module.search.run_case = guarded_run_case
            owner_module.OUT = OUT / condition
            owner_module.CASES = CASES
            owner_module.main()
            original_audit = OUT / condition / "blocked-owner-help-audit.json"
            audit = json.loads(original_audit.read_text())
            if len(guards) != len(CASES):
                raise AssertionError("Owner-help branch did not reach guard in every case")
            rows.extend({"condition": condition, **row}
                        for row in audit["rows"])
    finally:
        owner_module.HeldoutMovingRoom = original_room
        owner_module.search.run_case = original_run_case
        owner_module.OUT = original_out
        owner_module.CASES = original_cases
    report = {"scope": "V333 live selected Direct Frame blocked-route owner help after the V332 six-view actor hazard monitor and kinematic protective shoulder response. One seed has present, absent and identical-lookalike owner-script modes, each paired with a late moving or parked low prop. The owner is scripted and may instantaneously present an item; actor range/visual choices do not receive scene truth. This tests social state transitions after a hazard, not human-arm dynamics, tactile grasp or deployable sensor hardware.",
              "source_sha256": sha(SOURCE),
              "v317_source_sha256": sha(Path(owner_module.__file__)),
              "v332_guard_source_sha256": sha(Path(guard_module.__file__)),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "live-guard-owner-help-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
