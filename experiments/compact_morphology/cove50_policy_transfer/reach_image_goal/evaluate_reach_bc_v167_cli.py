"""Pass the frozen V95 native CLI to V167 and derive its aligned trace path."""

from pathlib import Path
import sys

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.evaluate_reach_bc_v167 import main


if __name__ == "__main__":
    if "--aligned-trace" in sys.argv or "--trace" not in sys.argv or "--student-actor" in sys.argv:
        raise ValueError("Expected V95 trace; wrapper fixes the V167 student checkpoint")
    trace = Path(sys.argv[sys.argv.index("--trace") + 1])
    checkpoint = Path(__file__).resolve().parents[4] / (
        "artifacts/ape-low-level-training-20261015/reach-bc-v167/checkpoint.pt")
    sys.argv.extend(["--aligned-trace", str(trace.with_name(trace.stem + "-aligned.npz")),
                     "--student-actor", str(checkpoint)])
    main()
