"""V301: owner-facing Direct Frame search with unobserved variable slip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
from run_lateral_clutter_shin_search_v284 import run_case
from run_localized_low_clutter_v300 import LocalizedLowClutterRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v301"
SEED = 138
HEIGHT_M = .04
MODES = ("present", "absent", "identical_lookalike")
POSE_SOURCES = ("command", "motion_checked_rgbd")
# Frozen before paired replay. This spans the 0.55–0.94 values exercised in
# the V297/V298 scripted pose route, but now the live planner chooses steps.
SLIPS = (.55, .92, .62, .85, .58, .94, .72, .88)


class VariableSlipRoom(LocalizedLowClutterRoom):
    def __init__(self, *, mode: str, pose_source: str, branch: Path) -> None:
        self.applied_slips: list[float] = []
        super().__init__(mode=mode, crate_height_m=HEIGHT_M,
                         pose_source=pose_source, branch=branch)

    def move(self, vx: float, vy: float, *, slip: float = .88,
             safety_probe=None) -> None:
        hidden_slip = SLIPS[len(self.applied_slips) % len(SLIPS)]
        self.applied_slips.append(hidden_slip)
        super().move(vx, vy, slip=hidden_slip, safety_probe=safety_probe)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for mode in MODES:
        for source in POSE_SOURCES:
            branch = OUT / source / f"seed{SEED}-{mode}"
            branch.mkdir(parents=True, exist_ok=True)
            holder = []

            def factory(seed: int, requested_mode: str):
                if seed != SEED or requested_mode != mode:
                    raise ValueError("Unexpected room request")
                room = VariableSlipRoom(mode=mode, pose_source=source,
                                        branch=branch)
                room.layout(seed, absent=mode != "present", furnished=True)
                holder.append(room)
                return room

            result = run_case(SEED, mode, vision, exemplar, owner_hist,
                              room_factory=factory, output_root=OUT / source,
                              sensor_case_prefix="v301-shared")
            trace = branch / "localization-trace.json"
            trace.write_text(json.dumps({
                "mode": mode, "pose_source": source,
                "slips_evaluator_only": holder[0].applied_slips,
                "events": holder[0].localization_events}, indent=2) + "\n")
            row = {"mode": mode, "pose_source": source,
                   "status": result["status"],
                   "owner_questions": result["owner_questions"],
                   "motion_steps": len(holder[0].applied_slips),
                   "evaluator_min_model_clearance_m":
                       result["evaluator_min_model_clearance_m"],
                   "evaluator_pose_error_m": result["evaluator_pose_error_m"],
                   "evaluator_wrong_candidate_views":
                       result["evaluator_wrong_candidate_views"],
                   "route_events": [event for event in result["events"]
                                    if event["action"] in (
                                        "lateral_sector", "return_sector")],
                   "localization_statuses": [event["status"]["status"]
                                            for event in holder[0].localization_events],
                   "localization_gates": [event.get("gate")
                                          for event in holder[0].localization_events[1:]],
                   "localization_trace_path": str(trace.relative_to(ROOT)),
                   "localization_trace_sha256": sha(trace),
                   "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                   "video_sha256": result["video_sha256"]}
            rows.append(row)
            (OUT / "variable-slip-search-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(mode, source, row["status"], row["motion_steps"],
                  row["evaluator_pose_error_m"],
                  row["evaluator_min_model_clearance_m"], flush=True)
    report = {"scope": "V301 paired selected Direct Frame seed138 true 4cm floor-clutter live visual owner search under alternating hidden 0.55–0.94 translation slip. Command-only versus frozen V298 motion-checked half-weight upper RGB-D pose fusion with same deterministic 1cm depth noise/15% holes and matched four-shin range noise. V284 point-only route gate, DINO/hue item association and owner exact-instance response unchanged. Evaluator true pose, minimum full-body gap and imposed slip score only. Static one-room kinematic surrogate; no moving obstacle, native stop, calibrated sensors, uncertainty or occupied/free/unknown volume proof.",
              "source_sha256": sha(SOURCE),
              "v300_source_sha256": sha(
                  ROOT / "experiments/ape_high_level/closed_loop/run_localized_low_clutter_v300.py"),
              "selected_visual_sha256": sha(selected_visual()),
              "slip_schedule_evaluator_only": SLIPS, "rows": rows}
    (OUT / "variable-slip-search-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
