"""Probe a fresh taught-item view from the monitored eastern waiting pose."""

from __future__ import annotations

import json
from pathlib import Path

import run_layout2_cooperative_east_show_v779 as east
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_return_v599 as memory


SOURCE = Path(__file__)
OUT = east.probe.side.ROOT / "artifacts/ape-robocasa-layout2-held-station-view-v781"


def main(*, out=OUT, probe_min_raw_range_m=.55, recheck=False,
         turned_factory=None, response_probe=False, emergency=False,
         directional_emergency=False, worker_range_dropout_s=None):
    out.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-held-station-view-probe.v1",
        "source_sha256": route.sha256(SOURCE),
        "reference_runner_sha256": route.sha256(east.SOURCE),
        "probe_at_episode_time_s": 16.,
        "profiles": ["static", "capture_triggered_direct_then_northwest_turn"],
        "actor_gate": "fresh range localization, >=20cm departure, raw return threshold, 2s finite held-track screen, no prior range dropout",
        "raw_return_threshold_m": probe_min_raw_range_m,
        "camera_inference": "head/chest item view with 20ms raw guard and 10Hz range packets during image worker",
        "probe_only": not recheck,
        "held_view_recheck": recheck,
        "actor_response_probe_only": response_probe,
        "actor_emergency_response": emergency,
        "actor_initial_radial_away_gate": directional_emergency,
        "worker_range_dropout_elapsed_s": worker_range_dropout_s,
        "claim_requires": "fresh second frame, tracked same box, metric item-memory agreement, healthy range pose and no worker/held warning" if recheck else
            "no item claim from probe"}
    (out / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old_out, old_memory_out, old_callback = east.OUT, memory.OUT, memory.callback

    def callback_with_held_view(**kwargs):
        return old_callback(held_view_probe_at_s=16.,
                            held_view_min_raw_range_m=probe_min_raw_range_m,
                            held_view_recheck=recheck,
                            held_view_response_probe=response_probe,
                            held_view_emergency=emergency,
                            held_view_directional_emergency=
                                directional_emergency,
                            held_view_worker_range_dropout_s=
                                worker_range_dropout_s,
                            **kwargs)

    try:
        east.OUT = out
        memory.OUT = out / "memory-packets"
        memory.callback = callback_with_held_view
        east.main(turned_factory=(turned_factory or east.turn.ExitTurnProp))
        for label in ("static", "turned"):
            path = out / label / "report.json"
            report = json.loads(path.read_text())
            report["held_view_declaration_sha256"] = route.sha256(
                out / "predeclared.json")
            path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        east.OUT, memory.OUT, memory.callback = old_out, old_memory_out, old_callback


if __name__ == "__main__":
    main()
