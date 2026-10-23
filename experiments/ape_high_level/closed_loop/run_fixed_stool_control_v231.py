"""V231: keep V229's introduced stool fixed at its starting pose."""

from __future__ import annotations

import json
from pathlib import Path

import run_moving_stool_short_depth_v229 as moving
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v231"


def main() -> None:
    old_out, old_source, old_stool_y = moving.OUT, moving.SOURCE, moving.stool_y
    moving.OUT = OUT
    moving.SOURCE = SOURCE
    moving.stool_y = lambda time_s: -.30
    try:
        moving.main()
    finally:
        moving.OUT, moving.SOURCE, moving.stool_y = old_out, old_source, old_stool_y
    geometry_path = OUT / "moving-stool-geometry-audit.json"
    geometry = json.loads(geometry_path.read_text())
    geometry["scope"] = ("V231 evaluator-only 20ms collision-geometry check for the "
                         "introduced but stationary stool at (0.60,-0.30)m against all "
                         "other props/furniture. Same initial changed-room placement as V229; "
                         "no moving stool trajectory.")
    for row in geometry["rows"]:
        row["stool_path_xy_m"] = [[.60, -.30]]
        row.pop("trajectory_duration_s", None)
    fixed_geometry_path = OUT / "fixed-stool-geometry-audit.json"
    fixed_geometry_path.write_text(json.dumps(geometry, indent=2) + "\n")
    geometry_path.unlink()
    raw = OUT / "moving-stool-short-depth-live-audit.json"
    result = json.loads(raw.read_text())
    result["scope"] = ("V231 controlled V229 comparison with the same newly introduced "
                       "stool held at (0.60,-0.30)m through all motion, look and wait "
                       "substeps. Same V225 short head depth, frozen RGB floor target, "
                       "origin route, virtual range/instant stop and selected-body kinematics. "
                       "Separates changed static geometry from stool movement, but not a "
                       "physical stool, depth device, low range or native stop.")
    result["source_sha256"] = sha(SOURCE)
    result["v229_source_sha256"] = sha(old_source)
    result["fixed_stool_geometry_sha256"] = sha(fixed_geometry_path)
    result.pop("moving_stool_geometry_sha256", None)
    (OUT / "fixed-stool-short-depth-live-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
