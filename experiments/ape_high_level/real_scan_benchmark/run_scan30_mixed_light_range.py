"""Cross camera-light change with a live range outage in complete owner tasks.

This composition keeps the two already frozen generic policies unchanged. It
is a condition-level stress test, not another room-specific recovery rule.
"""

from __future__ import annotations

from pathlib import Path

from freeze_scan20_shared_policy_sources import relative, sha
import run_scan30_lighting_cross as lighting
import run_scan30_range_recovery as recovery
import run_scan30_unified_policy as unified


HERE = Path(__file__).resolve().parent
ART = unified.ART
OUT = unified.ROOT / "assets/real_indoor_scenes/scan30_mixed_light_range_20261015"
PRECOMMIT = ART / "scan30_mixed_light_range_precommit.json"
REPORT = ART / "scan30_mixed_light_range_report.json"
RECOVERY_REFERENCE = recovery.REPORT
_base_freeze = recovery.freeze
_base_install = recovery.install_recovery
_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    cases = lighting.cohort()
    recovery.OUT = OUT
    recovery.PRECOMMIT = PRECOMMIT
    recovery.REPORT = REPORT
    recovery.HEALTHY = lighting.REPORT
    recovery.FAULT = RECOVERY_REFERENCE

    def freeze(selected: list[dict]) -> dict:
        definition = _base_freeze(selected)
        definition["schema"] = "ape.scan30-mixed-light-range-precommit.v1"
        definition["healthy_report_sha256"] = sha(lighting.REPORT)
        definition["fault_report_sha256"] = sha(RECOVERY_REFERENCE)
        definition["cross_lighting_precommit_sha256"] = sha(lighting.PRECOMMIT)
        definition["reference_range_precommit_sha256"] = sha(
            ART / "scan30_range_recovery_precommit.json")
        definition["policy_sources"] += [
            {"path": relative(HERE / "run_scan30_lighting_cross.py"),
             "sha256": sha(HERE / "run_scan30_lighting_cross.py")},
            {"path": relative(Path(__file__)), "sha256": sha(Path(__file__))}]
        definition["policy"] = (
            "Run the same from-start Scan30 owner teach/find/approach/review "
            "policy in all thirty rooms. Flip normal versus cool/dim search "
            "and route camera lighting from its matched healthy room. After "
            "the first accepted translated step, interrupt the independent "
            "two-height 10 Hz range source for one second; stop, wait for "
            "three fresh held-pose packets, renew owner-confirmed RGB-D item "
            "evidence, then allow one newly guarded route continuation and "
            "fresh final review. No per-room parameters or recovery tuning.")
        definition["score"] = ["whole-task result against both single-change "
            "references", "wrong offers and owner help", "stop/relook/resume "
            "on each observed gap", "sensor/pose traces and first-person video"]
        definition["limits"] = (
            "Thirty development-exposed static captured rooms, scripted owner "
            "and props, virtual range and ideal camera geometry. Historical "
            "rejected-body pose-shaped increments and 17 cm proxy are not a "
            "selected 50 cm robot. Lighting flips only search/route imagery, "
            "not the owner teaching view. This is a combined-condition software "
            "test, not a physical sensor or new-room qualification.")
        return definition

    def install(policy, show) -> None:
        _base_install(policy, show)
        lighting.install_cross_reshow(show)

    unified.cohort = lambda: cases
    recovery.freeze = freeze
    recovery.install_recovery = install
    _configured = True


def main() -> None:
    configure()
    recovery.main()


if __name__ == "__main__":
    main()
