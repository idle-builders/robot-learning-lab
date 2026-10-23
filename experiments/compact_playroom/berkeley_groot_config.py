"""GR00T N1.7 custom-embodiment mapping for the Berkeley button teacher data."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["external_view"]),
    "state": ModalityConfig(
        delta_indices=[0], modality_keys=["body_motion", "left_arm", "previous_action"]
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["walk", "left_arm"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
