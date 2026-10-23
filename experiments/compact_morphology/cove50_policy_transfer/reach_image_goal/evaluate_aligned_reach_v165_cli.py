"""Pass V95's frozen native CLI through to V165 with an aligned trace path."""

from pathlib import Path
import sys

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.evaluate_aligned_reach_v165 import main


if __name__ == "__main__":
    if "--aligned-trace" in sys.argv or "--trace" not in sys.argv:
        raise ValueError("Expected one V95 trace path; wrapper derives a separate aligned trace")
    trace = Path(sys.argv[sys.argv.index("--trace") + 1])
    sys.argv.extend(["--aligned-trace", str(trace.with_name(trace.stem + "-aligned.npz"))])
    main()
