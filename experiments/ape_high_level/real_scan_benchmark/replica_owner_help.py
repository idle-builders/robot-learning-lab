"""One owner-point response to an uncertain photographic Replica search.

The scripted owner uses evaluator identity only to choose a pixel. The actor
receives the point and chooses among proposals and metric depths that it had
already observed. Silence or a rejected point never supplies a motion goal.
"""

from __future__ import annotations

import json
from pathlib import Path


def actor_select(click: tuple[float, float], proposals: list[dict]) -> dict | None:
    x, y = click
    eligible = [proposal for proposal in proposals
        if proposal.get("actor_map_xy_m") is not None
        and proposal.get("local_depth", {}).get("status") == "located"
        and not proposal.get("border_crop")
        and proposal["box"][0] <= x <= proposal["box"][2]
        and proposal["box"][1] <= y <= proposal["box"][3]]
    return min(eligible, key=lambda proposal: (
        proposal["box"][2] - proposal["box"][0]) *
        (proposal["box"][3] - proposal["box"][1]), default=None)


def choose_response(search_trace: Path, root: Path, mode: str) -> dict:
    """Replay one scripted response from captured, actor-visible search views.

    `accurate` and `wrong` choose a pixel as the simulated owner; the actor
    then uses only the pixel and saved proposals. The evaluator IoU is kept
    separately to score owner confirmation and never ranks actor proposals.
    """
    if mode not in ("accurate", "wrong", "silent"):
        raise ValueError(f"Unknown owner response: {mode}")
    if mode == "silent":
        return {"status": "owner_no_answer", "response_mode": mode,
                "selected_proposal": None}
    aggregate = json.loads(search_trace.read_text())
    # Prefer the most recent observed view. It is usually closer to the
    # current route, so a confirmed point does not demand needless retracing.
    for station in reversed(aggregate["stations"]):
        for search_pass in reversed(station["passes"]):
            trace_path = root / search_pass["trace"]
            trace = json.loads(trace_path.read_text())
            for observation in reversed(trace["observations"]):
                target_box = observation.get("evaluator_target_box_xyxy")
                if mode == "accurate":
                    if target_box is None:
                        continue
                    clicks = [((target_box[0] + target_box[2]) / 2.,
                               (target_box[1] + target_box[3]) / 2.)]
                else:
                    # Simulate a plausible mistaken point on a different
                    # already detected proposal. The actor still resolves it
                    # by exactly the same point-in-proposal rule.
                    clicks = [((proposal["box"][0] + proposal["box"][2]) / 2.,
                               (proposal["box"][1] + proposal["box"][3]) / 2.)
                              for proposal in observation["proposals"]
                              if proposal.get("actor_map_xy_m") is not None
                              and proposal.get("evaluator_target_iou", 0.) < .05]
                for click in clicks:
                    selected = actor_select(click, observation["proposals"])
                    if selected is None:
                        continue
                    overlap = float(selected.get("evaluator_target_iou", 0.))
                    if mode == "wrong" and overlap >= .05:
                        continue
                    return {"status": ("owner_confirmed_point" if overlap >= .25
                                       else "owner_rejected_point"),
                        "response_mode": mode,
                        "station": station["station"],
                        "pass": search_pass["name"],
                        "frame": observation["frame"],
                        "rgb": str((trace_path.parent / observation["rgb_file"])
                                   .relative_to(root)),
                        "depth": str((trace_path.parent / observation["depth_file"])
                                     .relative_to(root)),
                        "click_xy": [float(click[0]), float(click[1])],
                        "selected_proposal": selected,
                        "evaluator_selected_target_iou": overlap,
                        "owner_confirmed": overlap >= .25}
    return {"status": "no_pointable_proposal", "response_mode": mode,
            "selected_proposal": None}
