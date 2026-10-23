"""Cross the ape's memory-return corridor with a fixture-clear low prop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-memory-moving-return-v603"
SOURCE = Path(__file__)
START = [3., -1.65]
STAGE = [1.7, -3.65]
END = [.95, -3.65]
ENTRY_START_S = 11.
ENTRY_END_S = 18.2
CROSS_END_S = 19.6
POST_STOP_UNTIL_S = 19.8
ENTRY_WAYPOINTS = ((3., -1.65), (1.8, -1.65), (1.8, -3.55),
                   (1.7, -3.55), (1.7, -3.65))
PROFILES = (("range", "stationary"), ("rgbd", "stationary"),
            ("range", "moving_across_return"),
            ("rgbd", "moving_across_return"))


def prop_motion(label: str):
    if label == "stationary":
        return lambda _: np.asarray(START, dtype=float)
    waypoints = np.asarray(ENTRY_WAYPOINTS, dtype=float)
    segments = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(segments)]

    def xy(time_s: float) -> np.ndarray:
        if time_s <= ENTRY_START_S:
            return waypoints[0].copy()
        if time_s < ENTRY_END_S:
            distance = cumulative[-1] * (time_s - ENTRY_START_S) / (
                ENTRY_END_S - ENTRY_START_S)
            segment = min(len(segments) - 1, int(np.searchsorted(
                cumulative[1:], distance, side="right")))
            fraction = (distance - cumulative[segment]) / segments[segment]
            return waypoints[segment] + fraction * (
                waypoints[segment + 1] - waypoints[segment])
        fraction = min(1., max(0., (time_s - ENTRY_END_S) /
                                (CROSS_END_S - ENTRY_END_S)))
        return np.asarray(STAGE) + fraction * (
            np.asarray(END) - np.asarray(STAGE))

    return xy


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    # The early owner route sees the same fixture-clear prop at its original
    # location. After owner confirmation, one condition rolls it into the
    # return corridor along an independently checked continuous path.
    case = {"id": "return_crossing", "from_xy_m": START,
            "to_xy_m": START, "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-moving-return.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-moving-return-declaration.v1",
        "base_scene": case, "profiles": PROFILES,
        "moving_prop": {"entry_waypoints_xy_m": ENTRY_WAYPOINTS,
                        "entry_start_s": ENTRY_START_S,
                        "entry_end_s": ENTRY_END_S,
                        "cross_to_xy_m": END,
                        "cross_end_s": CROSS_END_S},
        "stationary_prop_xy_m": START,
        "post_stop_until_s": POST_STOP_UNTIL_S,
        "fixture_preflight": "MuJoCo sampled-grid path has >=5 cm fixture gap; audit independently rechecks continuous 20 ms path",
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "joined_source_sha256": memory.digest(Path(joined.__file__)),
        "config_sha256": memory.digest(joined.CONFIG),
        "prior_map_sha256": memory.digest(joined.PRIOR),
    }, indent=2) + "\n")
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old_out, old_memory_out, old_hook, old_margin, old_motion = (
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK,
        route.MARGIN_M, joined.motion)
    joined.OUT = OUT
    memory.OUT = OUT
    route.MARGIN_M = .15
    episodes = []
    try:
        for source, label in PROFILES:
            joined.motion = lambda _, label=label: prop_motion(label)

            def hook(*, source=source, label=label, **kwargs):
                return memory.callback(
                    source=source, item_state="retained", run_label=label,
                    post_stop_until_s=POST_STOP_UNTIL_S, **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = f"memory_{source}_retained_{label}"
            episode = joined.episode(case, response, True, prior, policy,
                                     vision, exemplar)
            episodes.append(episode)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = episode["memory_return"]
            print(source, label, episode["outcome"],
                  None if summary is None else summary["route_status"],
                  None if summary is None else summary["item_status"],
                  None if summary is None else summary["minimum_proxy_gap_m_evaluator_only"],
                  flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-moving-return-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "joined_source_sha256": memory.digest(Path(joined.__file__)),
        "episodes": episodes,
    }, indent=2) + "\n")
    for source in (SOURCE, memory.SOURCE, Path(joined.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
