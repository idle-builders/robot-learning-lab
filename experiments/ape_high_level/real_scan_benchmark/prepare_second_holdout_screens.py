"""Apply the unchanged real-capture geometry/appearance screen to a new pool."""

from __future__ import annotations

import json

import prepare_fresh_holdout_screens as screen


screen.PRECOMMIT = screen.ROOT / \
    "artifacts/ape-real-room-next-captures-20261014/second_seven_capture_holdout_precommit.json"
screen.BASE = screen.ROOT / \
    "assets/real_indoor_scenes/scansceneverse/second_holdout_20261015"
screen.REPORT = screen.ROOT / \
    "artifacts/ape-real-room-next-captures-20261014/second_seven_capture_screen_report.json"


def save_rows(rows: list[dict], *, appearance: bool) -> None:
    screen.REPORT.write_text(json.dumps({
        "schema": "ape.second-seven-capture-screen.v1",
        "precommit": screen.relative(screen.PRECOMMIT),
        "appearance_screened": appearance,
        "rows": rows}, indent=2) + "\n")


screen.save_rows = save_rows


if __name__ == "__main__":
    screen.main()
