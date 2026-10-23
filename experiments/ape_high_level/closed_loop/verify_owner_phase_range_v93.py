"""Check V93 actor alarms from saved noisy point packets, not scene labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_owner_phase_range_v93 import (OUT, SOURCE, THRESHOLDS_M_PER_0P2S,
                                         TIMES_S, VIEWS)
from run import ROOT, sha
from run_shin_intrusion_posture_v324 import actor_alarm


SOURCE_AUDIT = Path(__file__)


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    assert report["source_sha256"] == sha(SOURCE)
    assert len(report["rows"]) == 24
    checks = 2
    results = []
    for condition in ("moving", "parked"):
        for seed in (170, 172):
            document = json.loads((OUT / f"{condition}-seed{seed}-packets.json").read_text())
            frames = document["frames"]
            assert len(frames) == len(TIMES_S)
            checks += 1
            packets = []
            for frame, time_s in zip(frames, TIMES_S):
                assert frame["time_s"] == time_s
                path = ROOT / frame["actor_packet_path"]
                assert sha(path) == frame["actor_packet_sha256"]
                with np.load(path) as data:
                    assert float(data["capture_time_s"]) == time_s
                    assert data["estimated_pose_xyyaw"].shape == (3,)
                    packets.append({key: data[key].copy() for key in data.files})
                checks += 4
            for view in VIEWS:
                for threshold in THRESHOLDS_M_PER_0P2S:
                    alerts = []
                    for index in range(2, len(frames)):
                        older = packets[index-2][f"{view}_points_xyz_m"]
                        newer = packets[index][f"{view}_points_xyz_m"]
                        pose = packets[index]["estimated_pose_xyyaw"]
                        replay = actor_alarm(older, newer, pose)
                        assert replay == frames[index]["actor_alarm_by_view"][view]
                        if any(item["closing_0p2s_m"] is not None and
                               item["closing_0p2s_m"] >= threshold
                               for item in replay["side_returns"].values()):
                            alerts.append(frames[index]["time_s"])
                        checks += 1
                    expected = next(row for row in report["rows"]
                                    if row["condition"] == condition and
                                    row["seed"] == seed and
                                    row["view"] == view and
                                    row["threshold_m_per_0p2s"] == threshold)
                    first = min(alerts) if alerts else None
                    assert first == expected["first_actor_alert_s"]
                    assert expected["first_evaluator_below_5cm_s"] == (
                        16.8 if condition == "moving" else None)
                    checks += 2
                    results.append({"condition": condition, "seed": seed,
                                    "view": view, "threshold": threshold,
                                    "first_alert_s": first})
    for row in results:
        if row["view"] == "front_four" and row["threshold"] == .018:
            assert row["first_alert_s"] == (16.4 if row["condition"] == "moving"
                                             else None)
            checks += 1
    result = {
        "schema": "robot-learning-lab.ape-owner-phase-range-audit.v1",
        "passed": True, "checks": checks,
        "source_sha256": sha(SOURCE_AUDIT),
        "results": results,
        "scope": "Recomputes 0.2s anonymous range alarms from retained 10Hz actor point packets, verifies packet hashes and capture times, and checks reported first-alert times. Simulator gap is used only as evaluator timing. Thresholds are development choices on two related scenes, not held-out specificity or physical stop proof.",
    }
    path = OUT / "independent-audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path, checks, "checks passed")


if __name__ == "__main__":
    main()
