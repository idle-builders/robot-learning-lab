"""Source-check the four frozen modular-versus-composition native pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failures(report: dict) -> list[str]:
    checks = {
        "ended early": report["ended_early"] or report["steps"] != 1200,
        "insufficient walk distance": report["walk_distance_m"] < 0.06,
        "stop drift": report["stop_tail_xy_m"] is None or report["stop_tail_xy_m"] > 0.03,
        "squat depth or return": not report["squat_depth_return_pass"],
        "arm rise or return": not report["arm_rise_return_pass"],
        "forbidden hand/body contact": not report["no_hand_or_body_contact"],
        "support": not report["support_windows_pass"],
        "rated motor duty": not report["rated_duty_windows_pass"],
        "reset discontinuity": report["switch_qpos_gap"] is None or report["switch_qpos_gap"] >= 1e-9,
    }
    return [name for name, failed in checks.items() if failed]


def video_evidence(path: Path) -> dict:
    reader = imageio.get_reader(path)
    try:
        count = reader.count_frames()
        samples = [reader.get_data(i) for i in sorted(set((0, count // 2, count - 1)))]
    finally:
        reader.close()
    return {"sha256": sha(path), "frames": count,
            "shape": list(samples[0].shape),
            "sample_std": [round(float(frame.std()), 3) for frame in samples],
            "decoded_nonblank": count > 0 and all(float(frame.std()) > 5 for frame in samples)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    method = json.loads(args.method.read_text())
    rows = []
    for case in method["cases"]:
        seed = case["seed"]
        reports = {}
        traces = {}
        paths = {}
        videos = {}
        for kind in ("source", "portfolio"):
            stem = args.source_dir / f"{kind}-{seed}"
            report_path, trace_path, video_path = (stem.with_suffix(suffix) for suffix in (".json", ".npz", ".mp4"))
            reports[kind] = json.loads(report_path.read_text())
            traces[kind] = np.load(trace_path)
            paths[kind] = {"report": str(report_path), "report_sha256": sha(report_path),
                           "trace": str(trace_path), "trace_sha256": sha(trace_path),
                           "video": str(video_path)}
            videos[kind] = video_evidence(video_path)
            report = reports[kind]
            if report["body_sha256"] != method["source_body_sha256"]:
                raise ValueError(f"{kind} {seed}: body hash mismatch")
            if report["checkpoint_sha256"] != method["actor_sha256"]:
                raise ValueError(f"{kind} {seed}: actor hash mismatch")
            if report["seed"] != seed or report["rail_v"] != case["rail_v"] or report["delay_ms"] != case["delay_ms"]:
                raise ValueError(f"{kind} {seed}: case identity mismatch")
            if report["walk_velocity_m_s"] != case["walk_velocity_m_s"]:
                raise ValueError(f"{kind} {seed}: command mismatch")
        if reports["portfolio"]["router_sha256"] != method["router_sha256"]:
            raise ValueError(f"{seed}: router hash mismatch")
        if sha(Path(__file__).with_name("evaluate_portfolio.py")) != method["portfolio_evaluator_sha256"]:
            raise ValueError("local portfolio evaluator differs from frozen method")
        if sha(Path(__file__).parents[1] / "cove50_low_level_handoffs" / "evaluate_walk_squat_arms.py") != method["source_evaluator_sha256"]:
            raise ValueError("local source evaluator differs from frozen method")
        a, b = traces["source"], traces["portfolio"]
        n = min(len(a["applied_action"]), len(b["applied_action"]))
        initial_qpos_gap = float(np.max(np.abs(a["initial_qpos"] - b["initial_qpos"])))
        initial_qvel_gap = float(np.max(np.abs(a["initial_qvel"] - b["initial_qvel"])))
        rows.append({
            "seed": seed, "case": case,
            "source": {"steps": reports["source"]["steps"], "pass": reports["source"]["full_continuity_gate_pass"],
                       "failures": failures(reports["source"])},
            "portfolio": {"steps": reports["portfolio"]["steps"], "pass": reports["portfolio"]["full_continuity_gate_pass"],
                          "failures": failures(reports["portfolio"])},
            "initial_qpos_gap": initial_qpos_gap, "initial_qvel_gap": initial_qvel_gap,
            "matched_steps": n,
            "max_applied_action_gap": float(np.max(np.abs(a["applied_action"][:n] - b["applied_action"][:n]))),
            "max_post_qpos_gap": float(np.max(np.abs(a["post_qpos"][:n] - b["post_qpos"][:n]))),
            "paths": paths, "videos": videos,
        })
    result = {
        "schema": "robot-learning-lab.modular-low-level-portfolio-report.v1",
        "method_sha256": sha(args.method),
        "scope": "four fresh paired full native development starts; modular composition, no new optimizer updates or physical transfer",
        "pairs": len(rows),
        "source_passes": sum(row["source"]["pass"] for row in rows),
        "portfolio_passes": sum(row["portfolio"]["pass"] for row in rows),
        "classification_regressions": sum(row["source"]["pass"] and not row["portfolio"]["pass"] for row in rows),
        "classification_rescues": sum(not row["source"]["pass"] and row["portfolio"]["pass"] for row in rows),
        "exact_start_pairs": sum(row["initial_qpos_gap"] == 0 and row["initial_qvel_gap"] == 0 for row in rows),
        "all_videos_decoded_nonblank": all(video["decoded_nonblank"] for row in rows for video in row["videos"].values()),
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
