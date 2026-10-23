"""Held-out faster second turn against the frozen unqualified emergency rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_layout2_held_station_second_turn_v791 as challenge
import run_layout2_held_station_view_probe_v781 as probe
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = probe.east.probe.side.ROOT / "artifacts/ape-robocasa-layout2-held-station-fast-turn-v797"
FAST_END_AFTER_CAPTURE_S = 8.0


class FastSecondTurnProp(challenge.SecondTurnProp):
    def __call__(self, time_s):
        if self.capture_time_s is None:
            return super().__call__(time_s)
        elapsed = time_s - self.capture_time_s
        if elapsed <= challenge.SECOND_START_AFTER_CAPTURE_S:
            return super().__call__(time_s)
        fraction = np.clip((elapsed - challenge.SECOND_START_AFTER_CAPTURE_S) /
                           (FAST_END_AFTER_CAPTURE_S -
                            challenge.SECOND_START_AFTER_CAPTURE_S), 0., 1.)
        return (challenge.first_turn.END + fraction *
                (challenge.SECOND_END - challenge.first_turn.END))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-held-station-fast-turn.v1",
        "source_sha256": route.sha256(SOURCE),
        "frozen_policy_source_sha256": route.sha256(
            Path(probe.memory.__file__)),
        "reference_slow_audit_sha256": route.sha256(
            probe.east.probe.side.ROOT /
            "artifacts/ape-robocasa-layout2-held-station-emergency-v795/audit-v796.json"),
        "same_entry_and_endpoint": True,
        "second_turn_start_after_capture_s":
            challenge.SECOND_START_AFTER_CAPTURE_S,
        "second_turn_end_after_capture_s": FAST_END_AFTER_CAPTURE_S,
        "slow_second_turn_end_after_capture_s":
            challenge.SECOND_END_AFTER_CAPTURE_S,
        "policy": "same 16s held-view rule and actor-only unqualified emergency selector; no speed-specific retuning"}
    (OUT / "fast-turn-declaration.json").write_text(
        json.dumps(declaration, indent=2) + "\n")
    probe.main(out=OUT, probe_min_raw_range_m=.40, recheck=True,
               turned_factory=FastSecondTurnProp,
               response_probe=True, emergency=True)
    for label in ("static", "turned"):
        path = OUT / label / "report.json"
        report = json.loads(path.read_text())
        report["fast_turn_declaration_sha256"] = route.sha256(
            OUT / "fast-turn-declaration.json")
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
