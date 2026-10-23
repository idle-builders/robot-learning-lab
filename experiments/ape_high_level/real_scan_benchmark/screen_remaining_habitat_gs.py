"""Complete the frozen 27-scene Habitat-GS ape-height visual screen.

Each real splat is downloaded for a three-view preview and then removed to
keep this disposable POC inside local disk limits. Previews and source records
remain; any promising scene must be reacquired for a moving-route test.
"""

import json
import subprocess
import sys
from pathlib import Path

from screen_habitat_gs import ASSETS


SCENES = (2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 19, 20,
          23, 24, 25, 26, 30, 33, 34, 35, 42, 43, 44, 45, 51, 52)
SCRIPT = Path(__file__).with_name("screen_habitat_gs.py")
OUTPUT = ASSETS / "remaining_27_ape_height_screen.json"


def main():
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.is_file() else {}
    results = previous.get("results", {})
    for number in SCENES:
        key = f"scene{number:02d}"
        folder = ASSETS / "train" / key
        triptych = folder / "preview_triptych.jpg"
        source = folder / f"{key}.gs.ply"
        existed_before = source.is_file()
        if triptych.is_file() and (folder / "screen.json").is_file():
            result = {"status": "previewed", "triptych": str(triptych),
                      "source_record": str(folder / "screen.json")}
        else:
            attempt = subprocess.run([sys.executable, str(SCRIPT),
                                      "--scene-number", str(number)],
                                     capture_output=True, text=True)
            result = {"status": "previewed" if attempt.returncode == 0 else "failed",
                      "triptych": str(triptych) if triptych.is_file() else None,
                      "source_record": str(folder / "screen.json")
                      if (folder / "screen.json").is_file() else None,
                      "error_tail": attempt.stderr[-1200:] if attempt.returncode else None}
        if not existed_before and source.is_file():
            source.unlink()
            result["download_pruned_after_preview"] = True
        results[key] = result
        OUTPUT.write_text(json.dumps({
            "schema": "ape.habitat-gs-remaining-visual-screen.v1",
            "source": "https://huggingface.co/datasets/RukawaY/gs_scenes",
            "camera_height_above_nav_m": 0.42,
            "scene_ids": [f"scene{i:02d}" for i in SCENES],
            "qualification": "Three-view stationary ape-height appearance preview only; no moving-route, aligned depth or owner-task qualification.",
            "results": results,
        }, indent=2) + "\n")
        print(key, result["status"], flush=True)


if __name__ == "__main__":
    main()
