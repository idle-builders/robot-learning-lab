"""From-start 30-room task comparison under biased body-motion increments.

Only the item-approach odometry packet changes. The owner approach, teaching,
RGB-D search, independent range health, target staging and decision rule remain
the shared Scan30 policy. This tests a sensor role, not a physical IMU/joint fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from freeze_scan20_shared_policy_sources import relative, sha
import run_scan30_unified_policy as unified
from run_scan30_lighting_cross import require_inference_weights, verify_case


ROOT = unified.ROOT
ART = unified.ART
HERE = Path(__file__).resolve().parent
OUT = ROOT / "assets/real_indoor_scenes/scan30_pose_bias_20261015"
PRECOMMIT = ART / "scan30_pose_bias_precommit.json"
ARMS = ("biased_native_only", "biased_native_range_corrected")
BIAS_SCALE = .8
BASELINE = ART / "scan30_unified_policy_report.json"


class BiasedNativePose:
    """Keep contact/health packets, undercount accepted translation by 20%."""

    def __init__(self, source: str) -> None:
        from native_shaped_pose import NativeShapedPose
        self.base = NativeShapedPose(source)
        self.source_sha256 = self.base.source_sha256

    def advance(self, command: np.ndarray) -> dict:
        packet = self.base.advance(command)
        if not packet["accepted"]:
            return packet
        raw = packet["measured_delta_world_m"]
        return {**packet, "raw_measured_delta_world_m": raw,
            "measured_delta_world_m": (BIAS_SCALE * np.asarray(raw)).tolist(),
            "injected_odom_scale": BIAS_SCALE}


class NoRangePoseCorrection:
    def __init__(self, _angles: np.ndarray) -> None:
        pass

    def correct(self, _ranges: np.ndarray, _pose: np.ndarray,
                _yaw_deg: float) -> dict:
        return {"accepted": False, "reason": "range_pose_disabled_ablation",
                "correction_xy_m": [0., 0.], "inliers": 0}


def freeze(cases: list[dict]) -> dict:
    definition = unified.freeze(cases)
    definition["schema"] = "ape.scan30-pose-bias-precommit.v1"
    definition["policy_sources"] += [{"path": relative(path),
        "sha256": sha(path)} for path in (Path(__file__),
        HERE / "native_shaped_pose.py", HERE / "range_pose_corrector.py",
        HERE / "run_hidden_item_rearrangement.py",
        HERE / "run_real_capture_owner_loop.py",
        HERE / "run_real_capture_colour_bank_search.py",
        HERE / "run_post_confirmation_approach_v2.py")]
    definition["baseline_report_sha256"] = sha(BASELINE)
    definition["arms"] = list(ARMS)
    definition["injected_odom_scale"] = BIAS_SCALE
    definition["evaluator_standoff_band_m"] = [.45, .75]
    definition["policy"] = (
        "Run the same complete owner teach/find/absent/approach/final-review "
        "task in all thirty recorded rooms. After each accepted item-approach "
        "translation, multiply the historical contact-gated joint/IMU-shaped "
        "measured increment by 0.8, retaining its health and provenance. One "
        "arm disables adjacent two-height range scan pose correction; the "
        "other leaves it enabled. Both retain the independent 10 Hz near-body "
        "range source, owner and item images, source-health gates and identical "
        "task decision rules. Include scripted owner-reshow continuation. "
        "No room-specific parameters or feedback from evaluator pose.")
    definition["score"] = ["whole-task completion, wrong offers and help",
        "actual target gap at claimed standoff", "maximum evaluator pose error",
        "range correction acceptance and safe stops", "synchronized media"]
    definition["limits"] = (
        "All thirty rooms development-exposed; ideal aligned RGB-D, yaw and "
        "range geometry. Bias begins only on the item-approach segment; the "
        "owner approach remains unchanged. The historical contact/IMU-shaped "
        "increments are a simulator-side proxy from a rejected body, not "
        "measured current 50 cm hardware. Scan matching assumes known yaw "
        "and mostly static geometry. No physical sensor or mount is selected.")
    return definition


def configure_arm(arm: str) -> None:
    import run_scan20_shared_owner_policy as policy
    import run_scan10_recovery_policy as recovery
    import run_scan10_show_again_continuation as show
    import run_alt_pair_full_action_20 as show_action
    from range_pose_corrector import RangePoseCorrector

    output = OUT / arm
    unified.BASE_OUT = output / "base"
    unified.SHOW_OUT = output / "show"
    policy.OUT = unified.BASE_OUT
    recovery.OUT = unified.BASE_OUT
    show.PRIOR_OUT = unified.BASE_OUT
    show.OUT = unified.SHOW_OUT
    show.SOURCE_OWNER = unified.SHOW_OUT / "source_owner"
    policy.NativeShapedPose = BiasedNativePose
    show_action.NativeShapedPose = BiasedNativePose
    corrector = (NoRangePoseCorrection if arm == ARMS[0]
                 else RangePoseCorrector)
    policy.RangePoseCorrector = corrector
    show_action.RangePoseCorrector = corrector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = unified.cohort()
    definition = freeze(cases)
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Frozen pose-bias definition changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(cases), ARMS)
        return
    require_inference_weights()
    import run_scan20_shared_owner_policy as policy
    import run_scan10_recovery_policy as recovery
    import run_scan10_show_again_continuation as show
    recovery.install_recoveries(policy)
    for arm in ARMS:
        if args.arm and arm != args.arm:
            continue
        configure_arm(arm)
        rows = []
        launched = 0
        for case in cases:
            if args.scene and case["scene_key"] != args.scene:
                continue
            verify_case(case)
            path = OUT / arm / case["scene_key"] / "combined_summary.json"
            if path.is_file():
                row = json.loads(path.read_text())
            elif args.max_new and launched >= args.max_new:
                break
            else:
                try:
                    row = unified.attempt(case, policy, recovery, show)
                except Exception as error:
                    row = {"scene_key": case["scene_key"],
                        "capture_group": case["capture_group"],
                        "cohort": case["cohort"],
                        "photometry": case["photometry"],
                        "result": "runner_failed", "error": repr(error),
                        "error_traceback": traceback.format_exc()[-3000:]}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row, indent=2) + "\n")
                launched += 1
            rows.append(row)
            report = ART / f"scan30_pose_bias_{arm}_report.json"
            report.write_text(json.dumps({"schema":
                "ape.scan30-pose-bias-arm-report.v1",
                "precommit": relative(PRECOMMIT), "arm": arm,
                "processed": len(rows), "rows": rows}, indent=2) + "\n")
            print(arm, case["scene_key"], row["result"], flush=True)


if __name__ == "__main__":
    main()
