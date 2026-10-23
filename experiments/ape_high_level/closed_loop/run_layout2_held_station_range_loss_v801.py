"""Withhold the 10 Hz track stream during held-view recognition."""

from pathlib import Path

import run_layout2_held_station_view_probe_v781 as probe


SOURCE = Path(__file__)
OUT = probe.east.probe.side.ROOT / "artifacts/ape-robocasa-layout2-held-station-range-loss-v801"


def main():
    probe.main(out=OUT, probe_min_raw_range_m=.40, recheck=True,
               worker_range_dropout_s=(.3, 1.2))


if __name__ == "__main__":
    main()
