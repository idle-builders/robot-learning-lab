"""Foot FSR actuator pucks: the native STEP of each silicone puck that presses on an FSR 402 under the foot pad.

Each puck is a disc of the coupon's actuator diameter, cut from the selected 0.5 mm silicone sheet (Polymax 1001526,
60 Shore A), centred on its FSR site with its top on the pad's pocket floor. Sites come from the right foot in
robot/electronics/sensing/foot-contact-coupon.json (right_sites); the left foot is the mirror (x -> -x). The frame is
the STEP frame of the v17 sensor study (robot/electronics/sensing/studies/purchasable/build.py, which first wrote these
files); the body manifests place each STEP with their native_step_offset_mm.

  .venv/bin/python -B robot/electronics/sensing/fsr_pucks.py [--out DIR]

Output (recreated by this script): DIR/<side>_<site>_actuator_v17.step for R and L and each site,
default DIR artifacts/cove50-sensor-integration/v17/parts. The meshes beside them are not written here.

The files are written byte-identical to the v17 export, so they still hash to the native_step_sha256 the body manifests
recorded: the STEP header carries the v17 export time, and each product occurrence keeps the number OpenCascade gave it
in that export session (a per-process counter, 409..415; 412 was the right pocket pad).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build123d import Compound, Cylinder, export_step

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("fsr-pucks")   # configured values: robot/electronics/sensing/fsr_pucks.bom.yaml config:

ROOT = Path(__file__).resolve().parents[3]
COUPON = ROOT / "robot/electronics/sensing/foot-contact-coupon.json"
OUT = ROOT / "artifacts/cove50-sensor-integration/v17/parts"

SHEET_THICKNESS_MM = CFG["SHEET_THICKNESS_MM"]
TOP_Z_MM = CFG["TOP_Z_MM"]
V17_EXPORT_TIME = CFG["V17_EXPORT_TIME"]
V17_OCCURRENCE = CFG["V17_OCCURRENCE"]


def pucks():
    """{part name: puck solid} for both feet."""
    cfg = json.loads(COUPON.read_text())
    radius = cfg["stack"]["actuator_diameter_mm"] / 2
    bottom = TOP_Z_MM - SHEET_THICKNESS_MM
    out = {}
    for side in ("R", "L"):
        for site in cfg["right_sites"]:
            x, y = site["xy_mm"]
            x = x if side == "R" else -x
            out[f"{side}_{site['id']}_actuator_v17"] = Cylinder(radius, SHEET_THICKNESS_MM).translate((x, y, (bottom + TOP_Z_MM) / 2))
    return out


def write(name, solid, path):
    export_step(Compound(children=list(solid.solids()), label=name), str(path), timestamp=V17_EXPORT_TIME)
    occ = V17_OCCURRENCE[name.removesuffix("_actuator_v17")]
    text = re.sub(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('\d+'", f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('{occ}'", path.read_text(), count=1)
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, solid in pucks().items():
        path = a.out / f"{name}.step"
        write(name, solid, path)
        bb = solid.bounding_box()
        print(f"{name}: volume {solid.volume:.4f} mm3, bounds {[round(v, 4) for v in (*bb.min, *bb.max)]} -> {path}")


if __name__ == "__main__":
    main()
