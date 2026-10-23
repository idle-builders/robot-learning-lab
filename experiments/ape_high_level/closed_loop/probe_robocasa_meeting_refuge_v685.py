"""Actor-map feasibility probe before choosing a new owner meeting station.

This stops at the owner-turn handoff. It deliberately does not score future
scripted prop paths or claim that a map route preserves the owner view.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_multitrack_v485 as multi


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-meeting-refuge-preflight-v685"
OFFSETS = ((0., 0.), (0., -.3), (-.2, .3), (-.3, .4),
           (-.4, .5), (-.5, .6), (-.6, .6), (-.3, -.2),
           (.2, .3), (.3, .4))
REFUGE_OFFSETS = ((0., 0.), (-.2, .2), (-.3, .3), (0., .3),
                  (.2, .2), (-.2, -.2), (0., -.3))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = OUT / "predeclared.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.meeting-refuge-map-probe.v1",
        "meeting_offsets_xy_m": OFFSETS,
        "refuge_offsets_from_station_xy_m": REFUGE_OFFSETS,
        "purpose": "Test actor-map known-goal reachability for owner-show stations and a distinct refuge before running new full episodes. No evaluator-script future scoring or owner-view claim.",
        "source_sha256": route.sha256(Path(__file__)),
        "prior_map_sha256": route.sha256(joined.PRIOR),
    }, indent=2) + "\n")
    policy = json.loads(joined.CONFIG.read_text())
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old_out, old_turn, old_margin = joined.OUT, joined.owner_turn, route.MARGIN_M
    joined.OUT = OUT
    route.MARGIN_M = .15
    captured = {}

    def probe(response, model, data, renderer, actual, estimate,
              time_s, use_depth, vision, exemplar):
        grid = multi.CAPTURED[-1]
        start = estimate[:2].copy()
        tracks = grid.multi_history[-1]["tracks"]
        options = []
        for offset in OFFSETS:
            station = start + offset
            _, state, path = grid.plan(start, station)
            if offset == (0., 0.):
                state, path = "known_goal_current", [grid.cell(start)]
            refuges = []
            if state in ("known_goal", "known_goal_current"):
                for refuge_offset in REFUGE_OFFSETS:
                    refuge = station + refuge_offset
                    _, refuge_state, refuge_path = grid.plan(station, refuge)
                    if refuge_offset == (0., 0.):
                        refuge_state, refuge_path = "known_goal_current", [grid.cell(station)]
                    refuges.append({
                        "offset_xy_m": refuge_offset,
                        "target_xy_m": refuge.round(5).tolist(),
                        "actor_map_state": refuge_state,
                        "path_xy_m": [grid.xy(cell).round(5).tolist()
                                      for cell in refuge_path],
                    })
            options.append({
                "offset_xy_m": offset,
                "target_xy_m": station.round(5).tolist(),
                "actor_map_state": state,
                "path_xy_m": [grid.xy(cell).round(5).tolist() for cell in path],
                "refuges": refuges,
            })
        captured.update({
            "capture_time_s": time_s,
            "estimated_xy_m": start.tolist(),
            "actual_xy_evaluator_m": actual.tolist(),
            "tracks_actor": tracks,
            "xs_m": grid.xs.tolist(),
            "ys_m": grid.ys.tolist(),
            "free_indices_yx": np.argwhere(grid.free).tolist(),
            "occupied_indices_yx": np.argwhere(grid.occupied).tolist(),
            "options": options,
        })
        return ("meeting_refuge_probe_only", [], [], None, None, time_s)

    joined.owner_turn = probe
    try:
        ep = joined.episode({"id": "meeting_refuge_probe",
                             "from_xy_m": crossing.START,
                             "to_xy_m": crossing.START,
                             "begin_s": 14., "end_s": 14.4},
                            "meeting_refuge_probe", True, prior,
                            policy, vision, exemplar)
    finally:
        joined.OUT, joined.owner_turn, route.MARGIN_M = (
            old_out, old_turn, old_margin)
    result = {"schema": "robot-learning-lab.meeting-refuge-map-probe-result.v1",
              "declaration_sha256": route.sha256(declaration),
              "route_outcome": ep["route_state"],
              "scene_fingerprint": ep["scene_fingerprint"],
              "actor_map": captured}
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in captured["options"]:
        supported = sum(x["actor_map_state"].startswith("known_goal")
                        for x in row["refuges"])
        print(row["offset_xy_m"], row["actor_map_state"],
              "known_refuges", supported, flush=True)


if __name__ == "__main__":
    main()
