"""Recheck the owner-taught item from a monitored eastern departure view."""

from pathlib import Path

import run_layout2_held_station_view_probe_v781 as probe


SOURCE = Path(__file__)
OUT = probe.east.probe.side.ROOT / "artifacts/ape-robocasa-layout2-held-station-recheck-v785"


def main():
    probe.main(out=OUT, probe_min_raw_range_m=.40, recheck=True)


if __name__ == "__main__":
    main()
