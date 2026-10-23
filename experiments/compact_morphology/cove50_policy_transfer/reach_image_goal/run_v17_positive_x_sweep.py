"""Extend the opened hard-miss goal sensitivity probe to +3/+4 mm X."""

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v17_hard_goal_bias as probe,
)

probe.DEST = probe.BASE / "v17/posthoc-positive-x-sweep"
probe.BIASES_MM = ((3.0, 0.0), (4.0, 0.0))

if __name__ == "__main__":
    probe.main()
