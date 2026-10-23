"""Fault RGB-D odometry mid owner-memory route; compare stop and range handoff."""

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
OUT = ROOT / "artifacts/ape-robocasa-memory-pose-failover-v601"
SOURCE = Path(__file__)
PROFILES = (
    ("retained", "none"),
    ("retained", "stop_on_rgbd_loss"),
    ("retained", "range_fallback"),
    ("removed_while_away", "range_fallback"),
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = next(row for row in policy["cases"] if row["id"] == "late_crossing")
    declaration = OUT / "predeclared-pose-failover.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-pose-failover-declaration.v1",
        "scene": case,
        "profiles": PROFILES,
        "fault": "drop the RGB-D depth pose packet after memory move 3",
        "handoff": "no further memory move until a newly captured 72-beam range scan registers",
        "item_states": "item retained or scripted owner removal at away viewpoint",
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "joined_source_sha256": memory.digest(Path(joined.__file__)),
        "config_sha256": memory.digest(joined.CONFIG),
        "prior_map_sha256": memory.digest(joined.PRIOR),
    }, indent=2) + "\n")
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old_out, old_memory_out, old_hook, old_margin = (
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, route.MARGIN_M)
    joined.OUT = OUT
    memory.OUT = OUT
    route.MARGIN_M = .15
    episodes = []
    try:
        for item_state, fault_profile in PROFILES:
            joined.POST_OWNER_HOOK = (
                lambda *, item_state=item_state, fault_profile=fault_profile, **kwargs:
                memory.callback(source="rgbd", item_state=item_state,
                                fault_profile=fault_profile, **kwargs))
            response = f"memory_rgbd_{item_state}"
            if fault_profile != "none":
                response += f"_{fault_profile}"
            episode = joined.episode(case, response, True, prior, policy,
                                     vision, exemplar)
            episodes.append(episode)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = episode["memory_return"]
            print(item_state, fault_profile, episode["outcome"],
                  None if summary is None else summary["route_status"],
                  None if summary is None else summary["item_status"], flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK = old_out, old_memory_out, old_hook
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-pose-failover-result.v1",
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
