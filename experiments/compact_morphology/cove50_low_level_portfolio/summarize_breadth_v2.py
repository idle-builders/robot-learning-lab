"""Source-check original V2 whole-start reports, traces and native videos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/modular-portfolio-breadth-v2"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    method = json.loads((BASE / "method.json").read_text())
    source_ok = all(digest(ROOT / path) == method["source_sha256"][name]
                    for name, path in method["sources"].items())
    if not source_ok:
        raise ValueError("Frozen source differs before summary")
    cases = []
    for case in method["cases"]:
        stem = BASE / "native" / f"seed-{case['seed']}-original"
        paths = {ext: stem.with_suffix(ext) for ext in (".json", ".npz", ".mp4", ".log")}
        if not any(path.exists() for path in paths.values()):
            cases.append({**case, "status": "not_run"})
            continue
        if not all(path.exists() for path in paths.values()):
            cases.append({**case, "status": "incomplete_original",
                          "present": {ext: path.exists() for ext, path in paths.items()}})
            continue
        report = json.loads(paths[".json"].read_text())
        with np.load(paths[".npz"], allow_pickle=False) as trace:
            trace_steps = {name: trace[name].shape[0] for name in trace.files
                           if name != "target_names" and name != "dt_s"
                           and name != "initial_qpos" and name != "initial_qvel"}
            initial_equal = bool(np.array_equal(trace["initial_qpos"],
                                                np.asarray(report["initial_qpos"])))
        count = 0
        min_std = None
        shape = None
        for frame in imageio.get_reader(paths[".mp4"]):
            count += 1
            if shape is None:
                shape = list(frame.shape)
            frame_std = float(np.std(frame))
            min_std = frame_std if min_std is None else min(min_std, frame_std)
        checks = {
            "seed": report["seed"] == case["seed"],
            "body": report["body_sha256"] == method["source_sha256"]["body"],
            "router": report["router_sha256"] == method["source_sha256"]["router"],
            "actors": all(report["checkpoint_sha256"][name] == method["source_sha256"][name]
                          for name in ("gait", "stop", "squat", "arms")),
            "case_parameters": (report["rail_v"] == case["rail_v"]
                                and report["delay_ms"] == case["delay_ms"]
                                and report["sole_friction"] == case["sole_friction"]
                                and report["walk_velocity_m_s"] == case["speed_m_s"]
                                and report["schedule_s"] == {
                                    "walk_end": case["walk_s"],
                                    "stop_end": case["walk_s"] + case["stop_s"],
                                    "squat_start": case["squat_start_s"],
                                    "squat_end": case["squat_start_s"] + case["squat_s"],
                                    "arm_start": case["arm_start_s"],
                                    "arm_end": case["arm_start_s"] + case["arm_s"]}),
            "steps_and_trace": all(value == report["steps"] for value in trace_steps.values()),
            "initial_qpos": initial_equal,
            "video_frames": count == (report["steps"] + 1) // 2,
            "video_nonblank": bool(min_std is not None and min_std > 1),
            "video_shape": shape == [480, 640, 3],
        }
        failures = []
        for reason, bad in (
            ("early_end", report["ended_early"] or report["steps"] != 1200),
            ("walk_travel", report["walk_distance_m"] < .06),
            ("stop_tail", report["stop_tail_xy_m"] is None
             or report["stop_tail_xy_m"] > .03),
            ("squat", not report["squat_depth_return_pass"]),
            ("arms", not report["arm_rise_return_pass"]),
            ("forbidden_contact", not report["no_hand_or_body_contact"]),
            ("support", not report["support_windows_pass"]),
            ("rated_nonwalk_duty", not report["rated_duty_windows_pass"]),
            ("switch_continuity", report["switch_qpos_gap"] is None
             or report["switch_qpos_gap"] >= 1e-9),
        ):
            if bad:
                failures.append(reason)
        if report["full_continuity_gate_pass"] != (not failures):
            checks["gate_consistency"] = False
        else:
            checks["gate_consistency"] = True
        motion_pass = not any(reason != "rated_nonwalk_duty" for reason in failures)
        cases.append({**case, "status": "source_checked" if all(checks.values()) else "source_fault",
                      "checks": checks, "steps": report["steps"],
                      "motion_pass_without_duty": motion_pass,
                      "full_gate_pass": report["full_continuity_gate_pass"],
                      "failure_reasons": failures,
                      "walk_distance_m": report["walk_distance_m"],
                      "stop_tail_mm": None if report["stop_tail_xy_m"] is None
                                      else report["stop_tail_xy_m"] * 1000,
                      "walk_leg_rated_duty_fraction": report["windows"].get("walk_end", {}).get(
                          "leg_duty_fraction"),
                      "report": str(paths[".json"].relative_to(ROOT)),
                      "trace": str(paths[".npz"].relative_to(ROOT)),
                      "video": str(paths[".mp4"].relative_to(ROOT)),
                      "report_sha256": digest(paths[".json"]),
                      "trace_sha256": digest(paths[".npz"]),
                      "video_sha256": digest(paths[".mp4"]),
                      "video_decoded_frames": count,
                      "min_decoded_frame_std": min_std})
    checked = [case for case in cases if case["status"] == "source_checked"]
    report = {"schema": "robot-learning-lab.modular-portfolio-breadth-v2.summary.v1",
              "claim_limit": "fresh development failure exposure of saved modular actors; no new training",
              "planned_families": len(method["cases"]), "source_checked_families": len(checked),
              "motion_passes_without_duty": sum(case["motion_pass_without_duty"] for case in checked),
              "full_gate_passes": sum(case["full_gate_pass"] for case in checked),
              "complete": len(checked) == len(method["cases"]),
              "method_sha256": digest(BASE / "method.json"), "source_ok": source_ok,
              "cases": cases}
    (BASE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in
                      ("planned_families", "source_checked_families", "motion_passes_without_duty",
                       "full_gate_passes", "complete")}))


if __name__ == "__main__":
    main()
