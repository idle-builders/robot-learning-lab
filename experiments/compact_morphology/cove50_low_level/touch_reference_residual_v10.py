"""V10 scene placement fix for the V9 observed-RGB residual touch task.

MJLab moves dynamic entities by each environment's grid origin, but the
static table entity in this task stays at the global origin. Independent
batched worlds therefore need coincident origins. Their physics states remain
separate; the shared static table geometry is present in every world.
"""

from touch_reference_residual_v9 import residual_touch_env_cfg as _v9_cfg


def residual_touch_env_cfg(*, cube_xy=None, cube_xy_range=None,
                           residual_gain: float = .08):
    cfg = _v9_cfg(cube_xy=cube_xy, cube_xy_range=cube_xy_range,
                  residual_gain=residual_gain)
    cfg.scene.env_spacing = 0.0
    return cfg
