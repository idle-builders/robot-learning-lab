"""Development replay of static/moving V718 packets through the actor tracker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from multi_novel_range_objects import MultiNovelRangeObjects
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_owner_inference_trigger_v716 as trial
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
PACKETS = trial.OUT / "range-coverage-v718.json"
OUT = trial.OUT / "tracker-replay-v719.json"


def replay(label, rows, prior, estimated_xy):
    tracker = MultiNovelRangeObjects(prior["xs"], prior["ys"],
                                     prior["free"], prior["occupied"])
    centers = []
    output = []
    streak = 0
    for row in rows:
        packet = row["paired_counterfactuals"]["fused_5_30cm"][
            f"{label}_72_beams_m"]
        snapshot = tracker.update(packet, estimated_xy, row["time_s"])
        present = [track for track in snapshot["tracks"] if track["status"] ==
                   "confirmed_now"]
        track = max(present, key=lambda item: item["hits"]) if present else None
        center = None if track is None else np.asarray(track["center_xy_m"])
        centers.append(center)
        speed = (float(np.linalg.norm((centers[-1] - centers[-3]) / .2))
                 if len(centers) >= 3 and centers[-1] is not None and
                 centers[-3] is not None else None)
        moving_evidence = speed is not None and speed > .45
        streak = streak + 1 if moving_evidence else 0
        output.append({"time_s": row["time_s"],
                       "actor_confirmed_track": track,
                       "actor_0p2s_center_speed_m_s": speed,
                       "development_two_scan_motion_streak": streak,
                       "development_interrupt": streak >= 2,
                       "raw_novel_return_count": snapshot["novel_return_count"]})
    first = next((item for item in output if item["development_interrupt"]), None)
    return {"first_development_interrupt_s": None if first is None else first["time_s"],
            "rows": output}


def main():
    packets = json.loads(PACKETS.read_text())
    episode = json.loads((trial.OUT / "report.json").read_text())["episode"]
    estimated_xy = np.asarray(episode["final_estimated_xy_m"])
    with np.load(transfer.PRIOR) as prior:
        rows = {label: replay(label, packets["rows"], prior, estimated_xy)
                for label in ("stationary", "moving")}
    result = {"schema": "robot-learning-lab.layout2-inference-tracker-replay.v1",
              "source_sha256": route.sha256(SOURCE),
              "range_packets_sha256": route.sha256(PACKETS),
              "prior_map_sha256": route.sha256(transfer.PRIOR),
              "estimated_pose_m": estimated_xy.tolist(),
              "rule_selected_after_viewing_this_development_pair": "0.2s track-center speed >0.45m/s for two consecutive 10Hz packets",
              "not_a_live_actor_response": True,
              "cases": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print({key: value["first_development_interrupt_s"] for key, value in rows.items()},
          flush=True)


if __name__ == "__main__":
    main()
