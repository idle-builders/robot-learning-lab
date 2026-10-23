"""V24: forecast only the requested route plus measured historical stop travel.

The skill deadline and native stop travel are actor-side command properties.
This remains a nominal root forecast, not a selected-body swept certificate.
"""

from __future__ import annotations

import json
from pathlib import Path

import online_cluster_guard as online
import online_corridor_trial as corridor
import run as guard
from kitchen_trial import sha


OUT = online.kitchen.OUT / "v24-plan-bounded"
SKILL_END_S = 3.0
HISTORICAL_STOP_ADVANCE_M = .11314
STOP_EQUIVALENT_S = HISTORICAL_STOP_ADVANCE_M / .35
OriginalOnlineGuard = online.OnlineClusterGuard


class PlanBoundedGuard(OriginalOnlineGuard):
    def update(self, packet: dict, pose, t: float) -> dict:
        skill_remaining_s = max(0., SKILL_END_S - t)
        horizon_s = min(guard.HORIZON_S,
                        skill_remaining_s + STOP_EQUIVALENT_S)
        old_horizon = guard.HORIZON_S
        guard.HORIZON_S = horizon_s
        try:
            result = super().update(packet, pose, t)
        finally:
            guard.HORIZON_S = old_horizon
        result["skill_remaining_s"] = round(skill_remaining_s, 5)
        result["forecast_horizon_s"] = round(horizon_s, 5)
        result["stop_tail_advance_budget_m"] = HISTORICAL_STOP_ADVANCE_M
        return result


def main() -> None:
    old_class = online.OnlineClusterGuard
    old_out = corridor.OUT
    online.OnlineClusterGuard = PlanBoundedGuard
    corridor.OUT = OUT
    try:
        corridor.main()
    finally:
        online.OnlineClusterGuard = old_class
        corridor.OUT = old_out
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.ape-robocasa-plan-bounded-guard.v1"
    report["method"] += (" V24 development-only action-contract change: cap"
                         " the nominal closest-approach horizon at remaining"
                         " requested forward-skill time plus .11314m of"
                         " historical native stop-tail travel at .35m/s.")
    report["actor_skill_contract"] = {
        "requested_forward_skill_end_s": SKILL_END_S,
        "planned_route_speed_mps": .35,
        "historical_stop_tail_advance_budget_m": HISTORICAL_STOP_ADVANCE_M,
        "stop_advance_equivalent_s": STOP_EQUIVALENT_S,
        "held_body_motion_monitor": "not_implemented",
    }
    report["plan_bounded_source_sha256"] = sha(Path(__file__))
    report["v23_report_sha256"] = sha(old_out / "report.json")
    report["development_on_v23_failures"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
