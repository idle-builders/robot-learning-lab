"""Selected R3 sensor/motor boundary; independent of historical actor mapping."""
from .boundary import BatchedRuntime, source_contract


def robot_cfg(**kwargs):
    from .mjlab_adapter import robot_cfg as build
    return build(**kwargs)


def get_runtime(env, entity_name="robot"):
    return env.scene[entity_name].actuators[0].boundary
