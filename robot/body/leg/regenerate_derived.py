#!/usr/bin/env python3
"""Regenerate cheap derived STL meshes or subset GLB views from retained native STEP/GLB evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from build123d import export_stl, import_step

ROOT = Path(__file__).resolve().parents[3]
ART_DIR = ROOT / "artifacts/robot/body/leg"


def regenerate_stl(step_path: Path, out_stl: Path | None = None) -> Path:
    target = out_stl or step_path.with_suffix(".stl")
    target.parent.mkdir(parents=True, exist_ok=True)
    solid = import_step(str(step_path))
    export_stl(solid, str(target), tolerance=0.04, angular_tolerance=0.15)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, help="Regenerate one STL from a retained STEP file")
    parser.add_argument("--output", type=Path, help="Output path for the regenerated STL")
    parser.add_argument("--all-stls", action="store_true", help="Regenerate all part STLs across variants")
    args = parser.parse_args()

    if args.step:
        out = regenerate_stl(args.step, args.output)
        print(f"Regenerated {out} ({out.stat().st_size} bytes)")
        return 0

    if args.all_stls:
        count = 0
        for step_file in sorted(ART_DIR.glob("*/parts/*.step")):
            out = regenerate_stl(step_file)
            count += 1
        print(f"Regenerated {count} STL files from retained STEP files.")
        return 0

    parser.error("Specify --step PATH or --all-stls")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
