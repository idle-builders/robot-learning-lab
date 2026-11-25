"""Package the ape-proportions body for the design dashboard: assembly GLB and front/side renders against the refit.

  .venv/bin/python -B robot/body/ape-proportions/package.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "motor-refit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body"))
from checks import bodylib as F  # noqa: E402  (body model: Body, load_mesh, src, ...)
from package import scene, project  # noqa: E402  (motor-refit packager)

REFIT = ROOT / "artifacts/robot/body/motor-refit/assembly-parts.json"
OUT = ROOT / "artifacts/robot/body/ape-proportions"


def main():
    man = json.load(open(OUT / "assembly-parts.json")); refit = json.load(open(F.src(REFIT)))
    scene(man).export(str(OUT / "assembly-with-full-shell.glb"))
    (OUT / "renders").mkdir(exist_ok=True)
    for shells in (True, False):
        fig, axs = plt.subplots(1, 4, figsize=(20, 9))
        project(refit, axs[0], "front", -5, 420, shells, "motor refit - front")
        project(man, axs[1], "front", -5, 420, shells, "ape proportions (arm x1.4/x1.2, foot x1.3) - front")
        project(refit, axs[2], "side", -5, 420, shells, "motor refit - side")
        project(man, axs[3], "side", -5, 420, shells, "ape proportions - side")
        plt.tight_layout(); name = f"ape-vs-refit-{'shells' if shells else 'mechanism'}.png"
        plt.savefig(OUT / "renders" / name, dpi=90); plt.close(fig); print("render", name)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    project(refit, axs[0], "front", -5, 60, True, "motor refit feet (front)")
    project(man, axs[1], "front", -5, 60, True, "x1.3 feet (front)")
    plt.tight_layout(); plt.savefig(OUT / "renders" / "feet-front.png", dpi=100); plt.close(fig); print("render feet")


if __name__ == "__main__":
    main()
