"""Summarize the frozen native Cove-50 COM-velocity support comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-com-velocity-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    for key in ("selected_design_pointer", "selected_body", "actuator_catalog", "probe_source"):
        pin = plan[key]
        assert sha(ROOT / pin["path"]) == pin["sha256"], key
    for collection in ("baseline_reports", "baseline_traces"):
        for side, pin in plan[collection].items():
            assert sha(ROOT / pin["path"]) == pin["sha256"], (collection, side)
    for line in (OUT / "trials/remote-sha256.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(ROOT / name) == digest, name

    rows = []
    for candidate in plan["cases"]:
        for side in ("right", "left"):
            name = f"{candidate['id']}-{side}-nominal"
            report_path = OUT / "trials" / f"{name}.json"
            trace_path = report_path.with_suffix(".npz")
            video_path = report_path.with_suffix(".mp4")
            report = json.loads(report_path.read_text())
            assert report["design_pointer_sha256"] == plan["selected_design_pointer"]["sha256"]
            assert report["body_sha256"] == plan["selected_body"]["sha256"]
            assert report["stance"] == side
            assert report["stance_com_velocity_horizon_s"] == candidate["velocity_horizon_s"]
            assert report["rail_v"] == 12 and report["delay_ms"] == 0
            assert report["slide_friction"] == 1 and report["seed"] == 7144
            assert video_path.is_file() and trace_path.is_file()
            first_rated = next((r for r in report["timeseries"] if r["rated_leg_joints"]), None)
            sign = 1 if side == "right" else -1

            def sample(time_s: float) -> dict:
                r = min(report["timeseries"], key=lambda row: abs(row["time_s"] - time_s))
                return {
                    "time_s": r["time_s"],
                    "com_toward_stance_mm": round(-sign * r["com_minus_stance_world_xy_m"][1] * 1000, 2),
                    "com_velocity_toward_stance_mm_s": round(-sign * r["filtered_com_lateral_velocity_m_s"] * 1000, 2),
                    "stance_load_fraction": r["stance_foot_load_fraction"],
                    "stance_slip_mm": round(r["stance_slip_m"] * 1000, 2),
                    "feet_grounded": r["feet_grounded"],
                }

            if candidate["id"] == "baseline-reload":
                prior = np.load(ROOT / plan["baseline_traces"][side]["path"])
                current = np.load(trace_path)
                for field in ("qpos", "qvel", "joint_target"):
                    assert np.array_equal(prior[field], current[field]), (side, field)

            rows.append({
                "case": name, "controller": candidate["id"], "stance": side,
                "velocity_horizon_s": candidate["velocity_horizon_s"],
                "duration_s": report["duration_s"],
                "completed_schedule": report["completed_schedule"],
                "exclusive_support_s": report["longest_single_support_s"],
                "rated_leg_fraction": report["rated_leg_fraction"],
                "max_stance_slip_mm": round(report["max_stance_slip_m"] * 1000, 2),
                "first_rated_time_s": first_rated["time_s"] if first_rated else None,
                "first_rated_joints": ", ".join(first_rated["rated_leg_joints"]) if first_rated else "",
                "nonsole_floor_frames": report["nonsole_floor_contact_frames"],
                "self_contact_frames": report["self_contact_frames"],
                "hand_contact_frames": report["hand_contact_frames"],
                "full_gate_pass": report["lift_skill_screen_pass"],
                "at_4_1_s": sample(4.1), "at_4_6_s": sample(4.6), "at_5_0_s": sample(5.0),
                "report": str(report_path.relative_to(ROOT)),
                "trace": str(trace_path.relative_to(ROOT)),
                "video": str(video_path.relative_to(ROOT)),
            })
    assert len(rows) == 6
    assert not any(row["full_gate_pass"] for row in rows)
    assert not list((OUT / "trials").glob("*-stress.json"))

    matrix = {"schema": "robot-learning-lab.cove50-com-velocity-support-matrix.v1", "cases": rows}
    (OUT / "summary-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    report = {
        "schema": "robot-learning-lab.cove50-com-velocity-support-outcome.v1",
        "objective": plan["objective"],
        "selected_design_pointer_sha256": plan["selected_design_pointer"]["sha256"],
        "selected_body_sha256": plan["selected_body"]["sha256"],
        "actuator_catalog_sha256": plan["actuator_catalog"]["sha256"],
        "native_runtime": "MJLab 1.3.0, MuJoCo 3.15.0, Torch 2.9.1+cu128, RTX 4090; pinned source and setup script retained. Runpod used for compute only and worker removed after results download.",
        "baseline_reproduced": "Both horizon-zero qpos, qvel and joint-target arrays exactly match the paired saved 2026-10-10 nominal traces.",
        "nominal_passes": {case["id"]: sum(row["full_gate_pass"] for row in rows if row["controller"] == case["id"]) for case in plan["cases"]},
        "gate": plan["full_gate"],
        "verdict": "Both velocity horizons fail both nominal sides. Damping limits stance-side COM overshoot and reduces rated-leg time but prevents sustained swing-foot unloading, then stance slip grows to 37.65-41.12 mm. No candidate reaches more than 0.02 s exclusive support; reserved stress was skipped under the frozen rule.",
        "interpretation": "At 4.1 s the baseline reaches 1.0 right-stance load and 16.9 mm stance-side COM offset, but rates right hip roll at 4.2 s and runs away. The 0.25/0.5 s horizons hold COM nearer 14-15 mm at 4.1 s; their stance load stays near 0.80, falls to about 0.73-0.74 by 5.0 s, and their stance sole drags beyond 15 mm. A lateral velocity term alone trades overload for lost contact and slip.",
        "scope_limit": "These are authored controllers using simulator COM and foot-force truth, not trained or deployable policies. The selected catalog motor and contact model are provisional. The diagnostic does not qualify support, foot lift, stepping or physical actuation.",
        "setup_note": "Initial bundle omitted two saved upright endpoint JSON files required by ape_walk import. Runtime install succeeded; those files were added to the bundle, remote SHA checked and import/version/CUDA reverified before any trial. The failed initial setup log is retained.",
        "evidence": {"plan": str((OUT / "plan.json").relative_to(ROOT)),
                     "matrix": str((OUT / "summary-matrix.json").relative_to(ROOT)),
                     "remote_hashes": str((OUT / "trials/remote-sha256.txt").relative_to(ROOT)),
                     "runner_log": str((OUT / "run-remote.log").relative_to(ROOT)),
                     "setup_log": str((OUT / "setup-remote.log").relative_to(ROOT))},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass_counts": report["nominal_passes"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
