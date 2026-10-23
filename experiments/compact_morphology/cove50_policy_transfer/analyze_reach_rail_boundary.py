"""Locate the motor phase that separates native 10 V and 11 V reach replays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-side-reach-20261009"


def window(path: Path, start: float, end: float) -> dict:
    rows = [row for row in json.loads(path.read_text())["timeseries"]
            if start < row["time_s"] <= end]
    force = np.asarray([row["leg_force_n_m"] for row in rows])
    # MJLab action and motor order: 0..4 left leg, 5..9 right leg; knee is 8.
    right_knee = force[:, 8]
    cap = float(right_knee.max())
    return {"samples": len(rows), "right_knee_peak_nm": round(cap, 3),
            "right_knee_near_peak_fraction": round(float(np.mean(right_knee >= .995 * cap)), 4),
            "any_leg_at_rated_fraction": round(float(np.mean([row["leg_at_rated"] for row in rows])), 4),
            "last_root_z_m": rows[-1]["root_xyz_m"][2],
            "last_root_tilt_deg": rows[-1]["root_tilt_deg"],
            "both_feet_fraction": round(float(np.mean([all(row["feet_grounded"]) for row in rows])), 4)}


def main() -> None:
    # Earlier authored native replays share seed, start, toy and target timing.
    eleven = ART / "native-heavy-lowgrip-11v-delay10.json"
    ten = ART / "native-stress-seed2144.json"
    r11 = json.loads(eleven.read_text())
    r10 = json.loads(ten.read_text())
    assert r11["body_sha256"] == r10["body_sha256"]
    assert r11["delay_ms"] == r10["delay_ms"] == 10
    assert r11["seed"] == r10["seed"]
    assert r11["start_xy_offset_m"] == r10["start_xy_offset_m"]
    summary = {
        "schema": "robot-learning-lab.cove50-reach-rail-boundary.v1",
        "body_sha256": r10["body_sha256"],
        "method": "native_MJLab_trace_comparison",
        "joint": "right_knee (leg action index 8)",
        "cases": [
            {"rail_v": 11, "seed": r11["seed"], "start_xy_offset_m": r11["start_xy_offset_m"],
             "contact_s": r11["longest_continuous_finger_toy_contact_s"],
             "duration_s": r11["duration_s"],
             "window_2_to_3_s": window(eleven, 2, 3),
             "window_3_to_4_s": window(eleven, 3, 4)},
            {"rail_v": 10, "seed": r10["seed"], "start_xy_offset_m": r10["start_xy_offset_m"],
             "contact_s": r10["longest_continuous_finger_toy_contact_s"],
             "duration_s": r10["duration_s"],
             "window_2_to_3_s": window(ten, 2, 3),
             "window_3_to_4_s": window(ten, 3, 4)},
        ],
        "interpretation": "In the matched seed-2144 native replays, the 10 V right knee sits at its provisional 1.308 Nm cap in 98% of the lean's 2-3 s load-transfer frames and the body tips by 3.50 s. At 11 V the knee reaches 1.438 Nm with 40% near-peak duty in that phase and holds 4.30 s finger contact. The new saved residual controller independently repeats the 11 V success and 10 V failure on reserved starts. This is a narrow modeled knee-support boundary, not a calibrated hardware voltage requirement or proof that no 10 V controller exists.",
        "sources": [str(eleven.relative_to(ROOT)), str(ten.relative_to(ROOT))],
        "saved_controller_replication": str((ART / "reach-residual-v1/evaluation/matrix.json").relative_to(ROOT)),
    }
    out = ART / "rail-boundary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
