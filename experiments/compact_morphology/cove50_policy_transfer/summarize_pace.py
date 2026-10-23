"""Compare measured gait, motor duty and stop behavior across Cove speed probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-recovery-20261008"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
STRESS_BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"
POINTER = ROOT / "robot/ape-current-design.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    body_sha = digest(BODY)
    specs = [
        ("effort100-fresh-forward-seed144.json", "Earlier actor · 0.35 m/s", "effort100-model745.pt"),
        ("effort100-slow02-stop-seed5201.json", "Earlier actor · 0.20 m/s · paired seed", "effort100-model745.pt"),
        ("effort100-slow015-stop12-seed4246.json", "Earlier actor · 0.15 m/s", "effort100-model745.pt"),
        ("lowpace99-forward02-seed5201.json", "Low-pace actor · 0.20 m/s · paired seed", "lowpace99-model844.pt"),
        ("lowpace99-stress02-seed6201.json", "Low-pace actor · 0.20 m/s · stress", "lowpace99-model844.pt"),
    ]
    cases = []
    for name, label, checkpoint_name in specs:
        path = OUT / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        expected_body = STRESS_BODY if "stress" in name else BODY
        if data["body_sha256"] != digest(expected_body):
            raise ValueError(f"Different body in {name}")
        worlds = data["worlds"]
        stop_worlds = [world.get("stop") for world in worlds]
        checkpoint = OUT / checkpoint_name
        cases.append({
            "case": label,
            "report": str(path.relative_to(ROOT)),
            "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.exists() else None,
            "checkpoint_sha256": digest(checkpoint) if checkpoint.exists() else None,
            "body": str(expected_body.relative_to(ROOT)),
            "seed": data["seed"],
            "worlds": len(worlds),
            "speed_m_s": data["velocity_command_m_s"][0],
            "duration_s": data.get("episode_length_s", 8.0),
            "stop_at_s": data["stop_at_s"],
            "one_meter_alternating": data["one_meter_with_alternating_steps"],
            "survived": sum(world["ended_by_timeout"] for world in worlds),
            "mean_forward_m": round(sum(world["forward_m"] for world in worlds) / len(worlds), 3),
            "mean_abs_lateral_m": round(sum(abs(world["lateral_m"]) for world in worlds) / len(worlds), 3),
            "mean_rated_leg_fraction": round(sum(world["fraction_frames_with_saturated_leg"] for world in worlds) / len(worlds), 3),
            "mean_moving_rated_leg_fraction": (
                round(sum(world["fraction_moving_frames_with_saturated_leg"] for world in worlds) / len(worlds), 3)
                if all("fraction_moving_frames_with_saturated_leg" in world for world in worlds)
                else (round(sum(world["fraction_frames_with_saturated_leg"] for world in worlds) / len(worlds), 3)
                      if data["stop_at_s"] is None else None)),
            "max_hand_force_n": max(world["hand_force_n"] for world in worlds),
            "max_body_contact_fraction": max(world["fraction_frames_with_body_contact"] or 0 for world in worlds),
            "mean_stop_travel_m": (round(sum(stop["travel_after_command_m"] for stop in stop_worlds) / len(worlds), 3)
                                   if all(stop_worlds) else None),
            "late_still_worlds": (sum(stop["late_double_support_fraction"] >= 0.9 and
                                      stop["late_mean_planar_speed_m_s"] <= 0.01 and
                                      not stop["touchdowns_in_last_second"] for stop in stop_worlds)
                                  if all(stop_worlds) else None),
            "featured": True,
        })
    report = {
        "schema": "robot-learning-lab.cove50-forward-pace-matrix.v1",
        "selected_pointer_sha256": digest(POINTER),
        "selected_body_sha256": body_sha,
        "rated_definition": "Fraction of frames with any leg actuator at or above 99.5% of the case's catalog-rated torque. Moving duty counts only frames before the zero-speed command; all-frame duty includes the stop interval.",
        "cases": cases,
    }
    (OUT / "pace-matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "one_meter": [case["one_meter_alternating"] for case in cases]}))


if __name__ == "__main__":
    main()
