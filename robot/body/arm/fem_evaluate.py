"""Shim: the arm study's FEM maths and load cases moved to robot/body/checks (fem_lib: mesher, elasticity, volume_quantiles;
fem check: per-part setups in data/part_settings.json) and to this folder's common.py (ARM_LOAD_WRENCHES, MATERIALS,
build_material_hooke). Re-exported for scripts that still `import fem_evaluate as FE` (ape-proportions/fem_arm.py). No check logic here."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from checks.fem_lib import volume_quantiles  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location("arm_common", HERE / "common.py")   # by path: other studies have their own common.py
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)
ARM_LOAD_WRENCHES, ELAST, MATERIALS, VON_MISES, build_material_hooke = (_c.ARM_LOAD_WRENCHES, _c.ELAST, _c.MATERIALS, _c.VON_MISES,
                                                                          _c.build_material_hooke)
