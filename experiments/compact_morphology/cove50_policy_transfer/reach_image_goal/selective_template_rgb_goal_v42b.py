"""Apply the V41 pixel rule at the physical reach entry after a head scan."""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.selective_template_rgb_goal_v41 import (
    SelectiveTemplateRgbGoal as V41SelectiveTemplateRgbGoal,
)


class SelectiveTemplateRgbGoal(V41SelectiveTemplateRgbGoal):
    """Keep the V41 image rule while allowing a delayed native reach entry."""

    template_entry_time_s = 7.5

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        delayed_entry = (self.template_entry_time_s != 7.5 and
                         abs(time_s - self.template_entry_time_s) <= 1e-6)
        # The inherited template matcher is restricted to time 7.5. Feed it
        # that entry marker only for the delayed capture; restore real time
        # immediately so goal age and evidence use the native clock.
        inherited_time = 7.5 if delayed_entry else time_s
        event, rgb = super().capture(native_qpos, inherited_time,
                                     pose_healthy=pose_healthy)
        if delayed_entry:
            event["time_s"] = round(time_s, 3)
            event["template_delayed_entry_s"] = round(time_s, 3)
            if event["used_for_control"]:
                self.last_seen_s = time_s
        return event, rgb
