"""Body-independent reset/observe/step boundary for high-level options.

The simulator supplies compact observations from perception and skill health.
Evaluator positions and capture identity are deliberately excluded here.
"""

from __future__ import annotations

from train_event_policy import decide


class OptionController:
    def __init__(self, policy: dict) -> None:
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self.pending: dict | None = None
        self.history: list[dict] = []

    def observe(self, phase: str, signals: dict) -> dict:
        if self.pending is not None:
            raise RuntimeError("Step the current observation before observing again")
        forbidden = {"scene_key", "capture_group", "target_true_xz",
                     "evaluator_actual_xz_m", "evaluator_actual_heading_deg"}
        if not phase or "phase" in signals or any(
            key in forbidden or key.startswith("evaluator_") for key in signals
        ):
            raise ValueError("High-level observation contains evaluator data")
        self.pending = {"phase": phase, **signals}
        return dict(self.pending)

    def step(self) -> dict:
        if self.pending is None:
            raise RuntimeError("Observe before requesting an option")
        decision = decide(self.policy, self.pending)
        self.history.append(decision)
        self.pending = None
        return decision
