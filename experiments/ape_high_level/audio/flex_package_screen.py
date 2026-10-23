"""Compare an upstream split microphone array with the Cove-50 head envelope.

This is a gross-width screen, not a mounting design. The Cove shell STEP is a
parallel visual study; the selected direct frame remains unchanged. In
particular, matching outer widths does not provide microphone ports, acoustic
isolation, cable bend, clearance, or an installed speaker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import build123d as bd


ROOT = Path(__file__).resolve().parents[3]
FLEX_DIR = ROOT / ".cache/ape-high-level/audio_candidates"
SHELL = ROOT / "artifacts/ape-frame50-cove-shell-v1/cove50-shell-neutral.step"
SHELL_REPORT = ROOT / "artifacts/ape-frame50-cove-shell-v1/report.json"
DESIGN = ROOT / "robot/ape-current-design.json"
FLEX_SOURCES = {
    "linear": (
        "respeaker_flex_linear.step",
        "https://files.seeedstudio.com/wiki/reSpeaker_flex/reSpeaker_xvf3800_flex_linear.step",
        "3f364668b4fe7e1e8e30b2491f0b7949b2cd6b06d96457f799f1ca6b589b399f",
    ),
    "core": (
        "respeaker_flex_core.step",
        "https://files.seeedstudio.com/wiki/reSpeaker_flex/reSpeaker_xvf3800_flex_Separate.step",
        "d1b3f7dc1df9836d9904716103cb9ea25a69176a677d4486d80f834d061a03ef",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extent(shape: bd.Shape) -> list[float]:
    box = shape.bounding_box()
    return [round(box.max.X - box.min.X, 2),
            round(box.max.Y - box.min.Y, 2),
            round(box.max.Z - box.min.Z, 2)]


def source_step(kind: str) -> Path:
    filename, url, expected_hash = FLEX_SOURCES[kind]
    path = FLEX_DIR / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".step.download")
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha256(temporary) != expected_hash:
            temporary.unlink()
            raise ValueError(f"Upstream {kind} STEP changed")
        temporary.replace(path)
    if sha256(path) != expected_hash:
        raise ValueError(f"Cached {kind} STEP changed")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    design = json.loads(DESIGN.read_text())
    shell_report = json.loads(SHELL_REPORT.read_text())
    if design["id"] != "cove-50-direct-frame":
        raise ValueError("The selected design changed")
    if shell_report["source_visual_body_sha256"] != design["evidence"]["visual_body_sha256"]:
        raise ValueError("The shell study does not match the selected visual body")
    if shell_report["shell_step_sha256"] != sha256(SHELL):
        raise ValueError("The saved shell changed")

    linear_path = source_step("linear")
    core_path = source_step("core")
    linear = bd.import_step(str(linear_path))
    core = bd.import_step(str(core_path))
    shell = bd.import_step(str(SHELL))
    parts = {part.label: part for part in shell.children}
    cranium = parts["head_cranium_front"]
    linear_size = extent(linear)
    core_size = extent(core)
    cranium_width = extent(cranium)[1]
    # The long board dimension lies across head-left/right in this comparison.
    total_width_margin = round(cranium_width - linear_size[0], 2)
    report = {
        "selected_design": design["id"],
        "source": {
            "linear_step": FLEX_SOURCES["linear"][1],
            "linear_sha256": sha256(linear_path),
            "core_step": FLEX_SOURCES["core"][1],
            "core_sha256": sha256(core_path),
            "shell_sha256": sha256(SHELL),
        },
        "board_extent_mm": {"linear_array": linear_size, "core": core_size},
        "saved_shell_front_cranium_width_mm": cranium_width,
        "selected_direct_head_cage_width_mm": 76.0,
        "linear_array_total_width_margin_inside_shell_mm": total_width_margin,
        "linear_array_each_side_if_centered_mm": round(total_width_margin / 2, 2),
        "interpretation": (
            "The 110 mm linear board nearly spans the saved 111.29 mm cranium. "
            "It exceeds the selected 76 mm bare head cage. This rules out "
            "an unmodified internal head-board assumption, not an external "
            "fixture or future custom acoustic package. Core can be elsewhere; "
            "connector, FPC, acoustic ports, tolerance and motion remain untested."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
