"""Run the same frozen learner option policy across transfer capture groups."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "real_scan_benchmark"))
import run_scan30_unified_policy as benchmark

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--auto-reshow", action="store_true")
    parser.add_argument("--cohort", choices=("development", "prior_transfer"),
                        default="prior_transfer")
    parser.add_argument("--fault", choices=("none", "pre_motion_range_gap",
                                       "owner_visual_pose_loss",
                                       "moving_person_final",
                                       "moving_person_final_range_gap",
                                       "blocked_route_walk"),
                        default="none")
    parser.add_argument("--min-free-mb", type=int, default=180)
    parser.add_argument("--route-sensor", choices=("separate_range",
                        "dual_rgbd"), default="separate_range")
    parser.add_argument("--monitor-coverage", choices=("all360", "front140"),
                        default="all360")
    parser.add_argument("--owner-clears-route", action="store_true")
    parser.add_argument("--scenario-variant", choices=("standard", "alt_pair"),
                        default="standard")
    args = parser.parse_args()
    cases = [c for c in benchmark.cohort() if c["cohort"] == args.cohort and
             (args.scene is None or c["scene_key"] in args.scene)]
    if not cases:
        raise ValueError("No scene matches the requested cohort")
    rows = []
    launched = 0
    report_path = OUT / f"{args.run}_batch_report.json"
    policy_sha = benchmark.sha(args.policy)
    for case in cases:
        if args.max_new and launched >= args.max_new:
            break
        output = OUT / case["scene_key"] / args.run
        summary = output / "summary.json"
        saved = json.loads(summary.read_text()) if summary.is_file() else None
        incomplete = bool(saved and saved.get("motion_authorized") and
                          saved.get("walk_result") is None)
        if saved is None or incomplete:
            if shutil.disk_usage(ROOT).free < args.min_free_mb * 1_000_000:
                print("Paused before disk exhaustion", flush=True)
                break
            command = [sys.executable, str(HERE / "live_option_episode.py"),
                "--scene", case["scene_key"], "--run", args.run,
                "--policy", str(args.policy), "--route-sensor",
                args.route_sensor, "--fault", args.fault,
                "--monitor-coverage", args.monitor_coverage,
                "--scenario-variant", args.scenario_variant]
            if args.recover:
                command.append("--recover")
            if args.owner_clears_route:
                command.append("--owner-clears-route")
            run = subprocess.run(command, cwd=ROOT,
                capture_output=True, text=True)
            launched += 1
            if run.returncode:
                rows.append({"scene_key": case["scene_key"],
                    "capture_group": case["capture_group"],
                    "result": "runner_failed", "error": run.stderr[-3000:]})
                print(case["scene_key"], "runner_failed", flush=True)
                report_path.write_text(json.dumps({
                    "schema": "ape.learner-option-transfer-batch.v1",
                    "policy": str(args.policy.resolve().relative_to(ROOT)),
                    "policy_sha256": policy_sha,
                    "runner_source_sha256": benchmark.sha(
                        HERE / "live_option_episode.py"),
                    "generic_recovery": args.recover,
                    "auto_reshow": args.auto_reshow,
                    "cohort": args.cohort, "fault": args.fault,
                    "scenario_variant": args.scenario_variant,
                    "route_sensor": args.route_sensor,
                    "monitor_coverage": args.monitor_coverage,
                    "owner_clears_route": args.owner_clears_route,
                    "processed": len(rows), "rows": rows}, indent=2) + "\n")
                break
        row = json.loads(summary.read_text())
        if not (output / "whole_episode_review.mp4").is_file():
            subprocess.run([sys.executable, str(HERE / "make_episode_reel.py"),
                            str(output)], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        entry = {"scene_key": case["scene_key"],
            "capture_group": case["capture_group"], "result": row["result"],
            "options": [e["executed"] for e in row["events"]],
            "video": str((output / "whole_episode_review.mp4").relative_to(ROOT)),
            "summary": str(summary.relative_to(ROOT))}
        if args.auto_reshow and row["result"] == "owner_or_teaching_unresolved":
            destination = f"{args.run}_reshow"
            continuation_dir = OUT / destination / case["scene_key"]
            continuation_summary = continuation_dir / "learner_summary.json"
            if not continuation_summary.is_file():
                command = [sys.executable, str(HERE / "continue_owner_reshow.py"),
                    "--scene", case["scene_key"], "--source-run", args.run,
                    "--destination", destination, "--policy", str(args.policy),
                    "--scenario-variant", args.scenario_variant]
                continuation_run = subprocess.run(command, cwd=ROOT,
                    capture_output=True, text=True)
                if continuation_run.returncode:
                    entry["continuation_error"] = continuation_run.stderr[-3000:]
            if continuation_summary.is_file():
                continuation = json.loads(continuation_summary.read_text())
                reel = continuation_dir / "whole_episode_review.mp4"
                if not reel.is_file():
                    subprocess.run([sys.executable,
                        str(HERE / "make_episode_reel.py"),
                        str(continuation_dir)], cwd=ROOT, check=True,
                        capture_output=True, text=True)
                entry["reshow_requested"] = continuation["requested"]
                entry["reshow_authorized"] = continuation["authorized"]
                entry["whole_result"] = continuation["result"]
                entry["reshow_video"] = str(reel.relative_to(ROOT))
                entry["reshow_summary"] = str(
                    continuation_summary.relative_to(ROOT))
        rows.append(entry)
        report_path.write_text(json.dumps({
            "schema": "ape.learner-option-transfer-batch.v1",
            "policy": str(args.policy.resolve().relative_to(ROOT)),
            "policy_sha256": policy_sha,
            "runner_source_sha256": benchmark.sha(HERE / "live_option_episode.py"),
            "generic_recovery": args.recover,
            "auto_reshow": args.auto_reshow,
            "cohort": args.cohort, "fault": args.fault,
            "scenario_variant": args.scenario_variant,
            "route_sensor": args.route_sensor,
            "monitor_coverage": args.monitor_coverage,
            "owner_clears_route": args.owner_clears_route,
            "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["result"], flush=True)
    print("report", report_path, len(rows), flush=True)


if __name__ == "__main__":
    main()
