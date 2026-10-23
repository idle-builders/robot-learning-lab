"""Use the existing two-confirmation visual tracker from the held station."""

from pathlib import Path

import run_layout2_held_station_view_probe_v781 as probe


SOURCE = Path(__file__)
OUT = probe.east.probe.side.ROOT / "artifacts/ape-robocasa-layout2-held-station-recheck-v787"


def main():
    probe.main(out=OUT, probe_min_raw_range_m=.40, recheck=True)


if __name__ == "__main__":
    main()
