"""Overlay small palm colliders on Berkeley's fixed hands for contact tasks.

The released USD has visual hand meshes and inertial hand links but no hand
colliders. This layer references the published USD and changes no joint or mass.
"""

import argparse
import os
from pathlib import Path

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    app = None
except ModuleNotFoundError:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from pxr import Gf, Usd, UsdGeom, UsdPhysics


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = args.source.resolve()
output = args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)

stage = Usd.Stage.CreateNew(str(output))
stage.GetRootLayer().subLayerPaths.append(os.path.relpath(source, output.parent))
root = stage.GetPrimAtPath("/berkeley_humanoid_lite")
if not root.IsValid():
    raise RuntimeError(f"Expected Berkeley robot prim in {source}")
stage.SetDefaultPrim(root)

for side in ("left", "right"):
    path = f"/berkeley_humanoid_lite/arm_{side}_hand_link/collisions/palm"
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(0.035)
    UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(-0.0025, 0.0135, -0.054))
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())

stage.GetRootLayer().Save()
print(output)
if app is not None:
    app.close()
