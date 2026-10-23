"""Audit the lowered raw-return held-view probe."""

import audit_layout2_held_station_view_v782 as base
import run_layout2_held_station_view_v783 as trial


def main():
    old_out, old_report = base.OUT, base.trial.OUT
    try:
        base.trial.OUT = trial.OUT
        base.OUT = trial.OUT / "audit-v784.json"
        base.main()
    finally:
        base.OUT, base.trial.OUT = old_out, old_report


if __name__ == "__main__":
    main()
