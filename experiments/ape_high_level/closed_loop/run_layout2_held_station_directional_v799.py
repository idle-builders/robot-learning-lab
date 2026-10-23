"""Development test of initial radial separation in the fast-turn emergency."""

from pathlib import Path

import run_layout2_held_station_fast_turn_v797 as fast
import run_layout2_held_station_view_probe_v781 as probe


SOURCE = Path(__file__)
OUT = probe.east.probe.side.ROOT / "artifacts/ape-robocasa-layout2-held-station-directional-v799"


def main():
    probe.main(out=OUT, probe_min_raw_range_m=.40, recheck=True,
               turned_factory=fast.FastSecondTurnProp,
               response_probe=True, emergency=True,
               directional_emergency=True)


if __name__ == "__main__":
    main()
