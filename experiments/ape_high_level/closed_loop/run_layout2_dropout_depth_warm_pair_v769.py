"""Compare depth-route policies after one matching camera/model warm-up turn."""

from __future__ import annotations

import json
from pathlib import Path

import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_adaptive_exit_turn_v749 as turn
import run_layout2_dropout_head_depth_response_v765 as best_effort
import run_layout2_dropout_head_depth_strict_v767 as strict
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = best_effort.dropout.trial.adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-dropout-depth-warm-pair-v769"


class Warmup(best_effort.DepthOffMonitor):
    output_branch = OUT / "warmup"


class BestEffort(best_effort.DepthOnMonitor):
    output_branch = OUT / "best_effort"


class Strict(strict.StrictDepthMonitor):
    output_branch = OUT / "strict"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-dropout-depth-warm-pair.v1",
        "profiles_in_order": ["warmup", "best_effort", "strict"],
        "same_prop": "V749 event-relative post-cue turn toward northwest exit",
        "same_0p8s_range_track_outage": True,
        "both_scored_profiles": "same 0.2s aimed head depth capture and source-health image invalidation; best_effort may dispatch unqualified radial goal, strict requires all V744 hypotheses supported",
        "comparison_limit": "model warmed in one process, but image/scan packets and capture times must be checked rather than assumed exact",
        "source_sha256": route.sha256(SOURCE),
        "best_effort_source_sha256": route.sha256(Path(best_effort.__file__)),
        "strict_source_sha256": route.sha256(Path(strict.__file__))}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    for label, monitor in (("warmup", Warmup),
                           ("best_effort", BestEffort),
                           ("strict", Strict)):
        adaptive.main(prop_factory=turn.ExitTurnProp, out=OUT / label,
                      profile=f"warmed depth response comparison {label}",
                      monitor_factory=monitor)
        path = OUT / label / "report.json"
        report = json.loads(path.read_text())
        report["warm_pair_declaration_sha256"] = route.sha256(OUT / "predeclared.json")
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
