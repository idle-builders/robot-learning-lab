"""Export the two ape contact-pad coupons as millimetre STL files.

These are flat TPU contact skins for the existing 90 x 60 mm sole and the
55 mm width sensitivity case. They are not the structural 24 mm foot.
"""

from pathlib import Path

import trimesh


OUT = Path(__file__).resolve().parent / "sole_coupons"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for width_mm in (60, 55):
        mesh = trimesh.creation.box(extents=(90.0, float(width_mm), 1.6))
        mesh.apply_translation((45.0, width_mm / 2, 0.8))
        mesh.export(OUT / f"ape_sole_pad_90x{width_mm}x1p6mm.stl")
        print(f"{width_mm} mm pad: {mesh.bounds.tolist()} mm")


if __name__ == "__main__":
    main()
