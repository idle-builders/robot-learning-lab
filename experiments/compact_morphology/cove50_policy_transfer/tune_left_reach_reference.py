"""Phase in a lateral left-hand reference offset on a frozen mirrored reach."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference-y-offset-m", type=float, default=-0.002)
    args = p.parse_args()
    if args.reference_y_offset_m != -0.002:
        raise ValueError("This frozen trial uses the predeclared -2 mm reference shift")
    saved = torch.load(args.source, map_location="cpu", weights_only=True)
    if (saved.get("hand") != "left" or saved.get("method") !=
            "bounded_observed_goal_pi_residual_on_native_teacher"):
        raise ValueError("Expected selected left mirrored feedback controller")
    reference = saved["center_ee_to_cube_xy_trajectory_m"].clone()
    if tuple(reference.shape) != (400, 2):
        raise ValueError("Unexpected feedback reference length")
    phase = torch.arange(400, dtype=reference.dtype) / 200
    phase = phase.clamp(0, 1)
    blend = phase * phase * (3 - 2 * phase)
    reference[:, 1] += args.reference_y_offset_m * blend
    tuned = dict(saved)
    tuned.update({
        "center_ee_to_cube_xy_trajectory_m": reference,
        "source_mirrored_checkpoint_sha256": sha(args.source),
        "reference_y_offset_m": args.reference_y_offset_m,
        "reference_shift_phase": "4 s smoothstep at 50 Hz",
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuned, args.out)
    print(f"Saved {args.out}; SHA-256 {sha(args.out)}")


if __name__ == "__main__":
    main()
