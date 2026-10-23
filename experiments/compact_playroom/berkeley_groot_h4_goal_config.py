"""Four-step Berkeley button mapping with measured target-relative state."""

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
        delta_indices=[0],
        modality_keys=[
            "button_relative", "hand_to_button", "body_motion", "left_arm", "previous_action"
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(4)),
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
