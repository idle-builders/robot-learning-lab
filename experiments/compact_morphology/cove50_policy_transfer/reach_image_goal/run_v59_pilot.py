"""Stage source-frozen V59 branch checks before its full native replay."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v59_combined_screen as v59,
)


def main() -> None:
    plan = v59.freeze()
    wanted = {"large-30", "large-36", "edge-01", "edge-12", "positive-10"}
    cases = [case for case in plan["cases"] if case["id"] in wanted]
    if len(cases) != len(wanted):
        raise ValueError("V59 pilot identities missing from frozen plan")
    v59.DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(v59.run_case, case, plan): case["id"] for case in cases}
        for future in as_completed(futures):
            row = future.result()
            print(json.dumps({key: row[key] for key in (
                "id", "seed", "pass", "contact_s",
                "negative_camera_sparse_partial_correction",
                "positive_camera_large_sparse_abstention", "edge_goal_bias_applied",
                "entry_goal_error_mm_scoring_only", "faults")}), flush=True)


if __name__ == "__main__":
    main()
