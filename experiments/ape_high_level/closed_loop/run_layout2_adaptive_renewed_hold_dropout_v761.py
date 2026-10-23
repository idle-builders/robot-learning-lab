"""Withhold early range-track updates in the frozen V757 turned-path challenge."""

from __future__ import annotations

import json
from pathlib import Path

import run_layout2_adaptive_renewed_hold_v757 as trial
import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_adaptive_exit_turn_v749 as turn
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = trial.adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-adaptive-renewed-hold-dropout-v761"
DROP_UNTIL_AFTER_CAPTURE_S = .8


class DropoutMonitor(trial.ReplanningEmergencyMonitor):
    def __init__(self, model, data, actual, script, grid, estimate):
        super().__init__(model, data, actual, script, grid, estimate)
        self.resumed = False

    def __call__(self, t):
        capture = self.script.capture_time_s
        assert capture is not None
        missing = t - capture < DROP_UNTIL_AFTER_CAPTURE_S
        if missing:
            self.next_scan_s = t + .1
        elif not self.resumed:
            self.next_scan_s = t
            self.resumed = True
        result = super().__call__(t)
        result["range_track_stream_health"] = "missing" if missing else "healthy"
        return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-adaptive-renewed-hold-dropout.v1",
        "frozen_trial_report_sha256": route.sha256(trial.OUT / "report.json"),
        "frozen_monitor_source_sha256": route.sha256(Path(trial.__file__)),
        "same_prop": "V749 post-cue turn toward V747 northwest exit",
        "missing_10hz_track_from_capture_until_s": DROP_UNTIL_AFTER_CAPTURE_S,
        "independent_20ms_raw_nearest_monitor_retained": True,
        "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    adaptive.main(prop_factory=turn.ExitTurnProp, out=OUT,
                  profile="frozen V757 monitor, missing first 0.8s range tracks",
                  monitor_factory=DropoutMonitor)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["dropout_declaration_sha256"] = route.sha256(OUT / "predeclared.json")
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
