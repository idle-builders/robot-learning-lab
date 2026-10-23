"""Repeat the frozen V757 turned-path episode to expose timing sensitivity."""

from __future__ import annotations

from pathlib import Path

import run_layout2_adaptive_renewed_hold_v757 as trial
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = trial.adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-adaptive-renewed-hold-repeat-v759"


if __name__ == "__main__":
    trial.main(out=OUT, profile="repeat V757 unchanged post-cue turn; independent camera-worker timing")
