"""Audit the combined lighting/range full-task condition and matched pairs."""

from __future__ import annotations

import json

import audit_scan30_range_recovery as audit
import run_scan30_mixed_light_range as mixed


def main() -> None:
    mixed.configure()
    audit.AUDIT = mixed.ART / "scan30_mixed_light_range_audit.json"
    audit.COMPARISON = mixed.ART / "scan30_mixed_light_range_comparison.json"
    audit.main()
    path = audit.COMPARISON
    result = json.loads(path.read_text())
    result["schema"] = "ape.scan30-mixed-light-range-comparison.v1"
    result["comparators"] = {
        "healthy_outcomes": "crossed lighting alone, same thirty rooms",
        "old_gap_outcomes": "range outage and recovery alone, same thirty rooms",
        "recovery_outcomes": "crossed lighting plus range outage/recovery"}
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "matched_rooms", "outage_exposed", "healthy_outcomes",
        "old_gap_outcomes", "recovery_outcomes", "issues")}, indent=2))


if __name__ == "__main__":
    main()
