"""Dispatch an explicitly unqualified known-free escape on a second warning."""

from pathlib import Path

import run_layout2_held_station_second_turn_v791 as challenge
import run_layout2_held_station_view_probe_v781 as probe


SOURCE = Path(__file__)
OUT = probe.east.probe.side.ROOT / "artifacts/ape-robocasa-layout2-held-station-emergency-v795"


def main():
    probe.main(out=OUT, probe_min_raw_range_m=.40, recheck=True,
               turned_factory=challenge.SecondTurnProp,
               response_probe=True, emergency=True)


if __name__ == "__main__":
    main()
