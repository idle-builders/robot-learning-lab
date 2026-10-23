"""Actor-only owner-stage option features and the independent action gate.

The option policy chooses whether to continue the bounded owner route or start
teaching. The route skill still supplies turns and short steps. Evaluator poses,
room identity, and truth are deliberately absent from the feature dictionary.
"""

from __future__ import annotations

import math


CONTINUE = "OWNER_CONTINUE"
TEACH = "OWNER_TEACH"


def _number(value, missing: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(result):
        return missing
    return max(-maximum, min(maximum, result))


def features(social: dict, *, pose_healthy: bool,
             known_low_beams: int, known_mid_beams: int,
             source_healthy: bool) -> dict:
    depth = _number(social.get("local_depth_m"), -1., 5.)
    bearing = abs(_number(social.get("owner_bearing_deg"), 180., 180.))
    return {"owner_visible": social.get("chosen_proposal") is not None,
        "owner_intent": str(social.get("intent", "unknown")),
        "owner_depth_m": depth, "owner_bearing_abs_deg": bearing,
        "nearest_forward_range_m": _number(
            social.get("nearest_forward_range_m"), -1., 5.),
        "known_low_beams": int(known_low_beams),
        "known_mid_beams": int(known_mid_beams),
        "pose_healthy": bool(pose_healthy),
        "source_healthy": bool(source_healthy)}


def viewpoint_supported(observation: dict) -> bool:
    depth = observation.get("owner_depth_m", -1.)
    return bool(observation.get("owner_visible") and
        observation.get("owner_intent") == "hold_near_person" and
        observation.get("owner_bearing_abs_deg", 180.) <= 20. and
        (1.20 <= depth <= 1.35 or 1.45 <= depth <= 1.65) and
        observation.get("nearest_forward_range_m", -1.) >= .9)


def safe_option(option: str, observation: dict) -> str:
    if not observation.get("pose_healthy") or not observation.get(
            "source_healthy") or observation.get("moving_obstacle"):
        return "HOLD"
    if option == TEACH:
        return TEACH if viewpoint_supported(observation) else CONTINUE
    return CONTINUE if option == CONTINUE else "HOLD"
