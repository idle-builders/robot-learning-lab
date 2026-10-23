"""Repeat the V627 slow crossing after a full cold-start warm-up episode."""

from pathlib import Path

import run_robocasa_memory_reactive_speed_v627 as trial


if __name__ == "__main__":
    trial.OUT = trial.ROOT / "artifacts/ape-robocasa-memory-reactive-warm-speed-v629"
    trial.SOURCE = Path(__file__).resolve()
    trial.PROFILES = ((1., False), (.7, False), (.7, True))
    trial.main()
