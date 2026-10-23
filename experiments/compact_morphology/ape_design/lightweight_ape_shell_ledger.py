"""Measure the sketch study's shell parts as provisional moving-link masses.

This intentionally reads the independent aesthetic CAD export rather than the
older ape_design/sketch_shell proxy. The link assignment is a design hypothesis:
it becomes a robot description only after the actual skeleton and axes exist.

Run from the repository root with .venv/bin/python.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import build123d as bd
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/lightweight-ape-20261004/source-shell-1mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/lightweight-ape-20261004/shell-link-ledger.json"
PLA_DENSITY_G_CM3 = 1.24  # Illustrative Bambu PLA density; not a measured print.


def link_for(name: str) -> str:
    if name.startswith(("L_", "R_")):
        side, kind = name.split("_", 1)
        if kind.startswith(("upper_arm", "elbow_cuff")):
            return f"{side}_upper_arm"
        if kind.startswith(("forearm", "wrist_cuff")):
            return f"{side}_forearm"
        if kind.startswith(("palm", "finger", "thumb")):
            return f"{side}_hand"
        if kind.startswith(("thigh", "knee_cuff")):
            return f"{side}_thigh"
        if kind.startswith(("shin", "ankle_cuff")):
            return f"{side}_shin"
        if kind.startswith(("foot", "toe", "sole")):
            return f"{side}_foot"
        if kind == "shoulder_trim":
            return "torso"
        if kind.startswith(("eye", "nostril")):
            return "head"
        raise ValueError(f"Unassigned shell part: {name}")
    if name.startswith(("torso", "chest_indicator", "neck_collar")):
        return "torso"
    if name.startswith("pelvis"):
        return "pelvis"
    if name.startswith(("head", "face", "muzzle", "mouth")):
        return "head"
    raise ValueError(f"Unassigned shell part: {name}")


def part_record(shape: bd.Shape) -> dict:
    mass = shape.volume * PLA_DENSITY_G_CM3 * 1e-6
    com = np.asarray(tuple(shape.center(bd.CenterOf.MASS)), dtype=float)
    inertia = np.asarray(shape.matrix_of_inertia, dtype=float) * PLA_DENSITY_G_CM3 * 1e-12
    return {"name": shape.label, "link": link_for(shape.label),
            "volume_mm3": shape.volume, "mass_kg": mass,
            "com_world_mm": com.tolist(), "inertia_at_com_world_kg_m2": inertia.tolist()}


def aggregate(items: list[dict]) -> dict:
    mass = sum(item["mass_kg"] for item in items)
    com = sum((item["mass_kg"] * np.asarray(item["com_world_mm"])
               for item in items), np.zeros(3)) / mass
    inertia = np.zeros((3, 3))
    for item in items:
        delta_m = (np.asarray(item["com_world_mm"]) - com) / 1000
        inertia += np.asarray(item["inertia_at_com_world_kg_m2"])
        inertia += item["mass_kg"] * (np.dot(delta_m, delta_m) * np.eye(3) - np.outer(delta_m, delta_m))
    return {"mass_kg": mass, "com_world_mm": com.tolist(),
            "inertia_at_com_world_kg_m2": inertia.tolist(),
            "part_names": [item["name"] for item in items]}


def main() -> None:
    assembly = bd.import_step(SOURCE)
    parts = [part_record(shape) for shape in assembly.children]
    if len(parts) != 67:
        raise RuntimeError(f"Expected the source study's 67 parts, got {len(parts)}")
    by_link = defaultdict(list)
    for item in parts:
        by_link[item["link"]].append(item)
    report = {
        "source_step": str(SOURCE.relative_to(ROOT)),
        "material_density_g_cm3": PLA_DENSITY_G_CM3,
        "status": "shell-only hypothetical link ownership; no inner skeleton, motors or joints",
        "mass_kg": sum(item["mass_kg"] for item in parts),
        "links": {name: aggregate(items) for name, items in sorted(by_link.items())},
        "parts": parts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"shell_mass_kg": report["mass_kg"],
                      "link_mass_g": {name: round(row["mass_kg"]*1000, 2)
                                      for name, row in report["links"].items()}}, indent=2))


if __name__ == "__main__":
    main()
