"""Challenge the unchanged whole owner policy with three simultaneous faults.

This reuses the frozen lighting, range-gap recovery and motion-bias mechanisms.
It adds no room-specific decision rule and keeps the independent range pose
correction enabled. The thirty rooms remain a development cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

from freeze_scan20_shared_policy_sources import relative, sha
import run_scan30_mixed_light_range as mixed
import run_scan30_pose_bias as pose
import run_scan30_range_recovery as recovery


HERE = Path(__file__).resolve().parent
ART = mixed.ART
OUT = mixed.unified.ROOT / "assets/real_indoor_scenes/scan30_compound_pose_light_range_20261016"
PRECOMMIT = ART / "scan30_compound_pose_light_range_precommit.json"
REPORT = ART / "scan30_compound_pose_light_range_report.json"
LIGHT_RANGE = mixed.REPORT
POSE_CORRECTED = ART / "scan30_pose_bias_biased_native_range_corrected_report.json"
_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    mixed.configure()
    base_freeze = recovery.freeze
    base_install = recovery.install_recovery
    recovery.OUT = OUT
    recovery.PRECOMMIT = PRECOMMIT
    recovery.REPORT = REPORT
    recovery.HEALTHY = LIGHT_RANGE
    recovery.FAULT = POSE_CORRECTED

    def freeze(cases: list[dict]) -> dict:
        definition = base_freeze(cases)
        definition["schema"] = "ape.scan30-compound-pose-light-range-precommit.v1"
        definition["policy_sources"] += [
            {"path": relative(path), "sha256": sha(path)}
            for path in (Path(__file__), HERE / "run_scan30_pose_bias.py")]
        definition["mixed_light_range_precommit_sha256"] = sha(mixed.PRECOMMIT)
        definition["pose_bias_precommit_sha256"] = sha(pose.PRECOMMIT)
        definition["injected_odom_scale"] = pose.BIAS_SCALE
        definition["evaluator_standoff_band_m"] = [.45, .75]
        definition["policy"] = (
            "Run one unchanged from-start owner teach/find/absent/approach/"
            "review task in all thirty recorded rooms. Flip each room's route "
            "and search image lighting, undercount every accepted item-approach "
            "motion increment by 20%, and interrupt the independent two-height "
            "range worker for one second after the first translated step. Keep "
            "the existing range pose correction, stop/reobserve/owner-confirm/"
            "resume rule, source-health gates and final review. No room-specific "
            "parameter, image selection, evaluator pose or new recovery rule.")
        definition["score"] = ["complete task and wrong confident offer",
            "actual evaluator target gap at claimed final standoff",
            "maximum evaluator capture pose error", "gap stop, renewed view "
            "and guarded continuation", "owner help and synchronized media"]
        definition["limits"] = (
            "Thirty development-exposed static captured rooms, scripted owner "
            "and props, virtual ideal RGB-D/range/yaw, historical rejected-body "
            "motion-shaped increments and 17 cm kinematic proxy. Bias begins "
            "only during item approach; owner approach stays unchanged. Scan "
            "matching assumes known yaw and mostly static geometry. This is a "
            "combined-condition software stress test, not new-room, learned "
            "policy, physical sensor or current-body qualification.")
        return definition

    def install(policy, show) -> None:
        import run_alt_pair_full_action_20 as action
        policy.NativeShapedPose = pose.BiasedNativePose
        action.NativeShapedPose = pose.BiasedNativePose
        base_install(policy, show)

    recovery.freeze = freeze
    recovery.install_recovery = install
    _configured = True


def main() -> None:
    configure()
    recovery.main()


if __name__ == "__main__":
    main()
