"""Posthoc saved-pose geometry for later opened V147 positive-target failures.

This reconstructs fingertip separation from saved CUDA qpos. Native contact
reports remain authoritative; no controller or contact simulation is rerun.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    analyze_v147_finger_clearance as geometry,
)


ROOT = geometry.ROOT
BASE = geometry.BASE
MATRIX = geometry.MATRIX
OUT = BASE / "later-positive-miss-finger-clearance.json"
FIGURE = BASE / "later-positive-miss-finger-clearance.png"
SEEDS = (24322, 24328, 24333, 24334)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(cases: list[dict]) -> None:
    colors = {"unchanged": "#3b6c8e", "neg2": "#178263", "pos2": "#d35440"}
    labels = {"unchanged": "Unchanged", "neg2": "−2 mm", "pos2": "+2 mm"}
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 11.5), sharex=True)
    for ax, case in zip(axes, cases):
        for variant in ("unchanged", "neg2", "pos2"):
            series = case["visits"][variant]["saved_pose_surface_distance_series_mm"]
            ax.plot([point[0] for point in series],
                    [point[1] for point in series],
                    color=colors[variant], label=labels[variant], linewidth=1.7)
        ax.axhline(0, color="#5e6264", linewidth=1, linestyle="--")
        ax.set_title(f"Training seed {case['seed']} · {case['observed_route']}")
        ax.set_ylabel("Surface gap (mm)")
        ax.set_ylim(-.8, 5)
        ax.grid(alpha=.18)
    axes[0].legend(loc="upper right", ncol=3, frameon=False)
    axes[-1].set_xlabel("Saved native trajectory time (s)")
    fig.suptitle("Opened +2 mm contact failures: saved-pose fingertip geometry",
                 fontsize=13)
    fig.text(.5, .01, "Below zero: geometric overlap. Native CUDA contact remains the outcome.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    model = geometry.model_with_diagnostic_cube()
    cases = []
    for seed in SEEDS:
        row = next(case for case in matrix["cases"] if case["seed"] == seed)
        if (row["choices"]["pos2"]["pass"] or
                not row["choices"]["neg2"]["pass"] or
                any(row["choices"][direction]["entry_parity_faults"]
                    for direction in ("neg2", "pos2"))):
            raise ValueError(f"Paired failure conditions changed: {seed}")
        paths = {
            "unchanged": (row["baseline_result"], row["baseline_trace"]),
            "neg2": (row["choices"]["neg2"]["result"],
                     row["choices"]["neg2"]["trace"]),
            "pos2": (row["choices"]["pos2"]["result"],
                     row["choices"]["pos2"]["trace"]),
        }
        cases.append({
            "seed": seed,
            "row_id": row["id"],
            "row_sha256": hashlib.sha256(json.dumps(
                row, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "observed_route": row["selection_mode"],
            "visits": {
                name: geometry.visit(model, ROOT / report, ROOT / trace)
                for name, (report, trace) in paths.items()
            },
        })
    draw(cases)
    output = {
        "schema": "robot-learning-lab.cove50-v147-later-finger-clearance.v1",
        "role": "posthoc_saved_cuda_trajectory_geometry_not_native_rescore",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "selection_disclosure": "Four later opened positive-target failures selected after their contact outcomes were seen.",
        "plan_sha256": matrix["plan_sha256"],
        "matrix_sha256_at_analysis": sha(MATRIX),
        "body": str(geometry.BODY.relative_to(ROOT)),
        "body_sha256": sha(geometry.BODY),
        "source_sha256": {"this": sha(Path(__file__)),
                          "geometry_reconstruction": sha(Path(geometry.__file__))},
        "analyzed_time_window_s": [13.0, 17.48],
        "time_alignment": ("Saved trace qpos is pre-step; report row i is post-step. "
                           "Reconstruct row i from qpos[i+1], omitting the final "
                           "row whose matching next pose is not retained."),
        "correction_note": ("Supersedes the initial same-index posthoc geometry "
                            "chart; its original JSON and PNG are retained under "
                            "v147/superseded-prestep-alignment/. Native CUDA "
                            "contact reports and trajectories did not change."),
        "figure": str(FIGURE.relative_to(ROOT)),
        "figure_sha256": sha(FIGURE),
        "cases": cases,
        "interpretation": (
            "The +2 mm targets at 24328, 24333 and 24334 have zero native "
            "fingertip contact frames and positive reconstructed clearance. "
            "At 24322 the fingertip reaches the cube but uninterrupted native "
            "contact lasts only 0.04 s; this is a contact-retention failure, "
            "not a simple clearance miss. All paired -2 mm targets make contact. "
            "The unchanged target at 24333 has zero contact, while its -2 mm "
            "target passes the full native gate at exactly 1.2 s. This is "
            "posthoc geometry of saved CUDA poses, not "
            "a local CPU contact rescore, learned selector or independent score."
        ),
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved later finger-clearance diagnostic: {len(cases)} opened starts; "
          f"sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
