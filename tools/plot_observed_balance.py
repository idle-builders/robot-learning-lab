"""Plot fixed versus observed trim at the same outward arm posture."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
for name in ("pip", "scout"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex="col", layout="constrained")
    fig.suptitle(f"{name.title()} · same outward posture, different lean reference")
    for col, case in enumerate(("tilt-recovery", "drive-return")):
        for mode, color, folder in [
            (
                "Fixed trim",
                "#bc663c",
                f"artifacts/{name}-composed-outward-selection-v1",
            ),
            (
                "Encoder-posture trim",
                "#187668",
                f"artifacts/{name}-composed-observed-nominal-balance-v1",
            ),
        ]:
            rows = [
                json.loads(line)
                for line in (Path(folder) / case / "trajectory.jsonl")
                .read_text()
                .splitlines()
            ]
            t = [r["time_s"] for r in rows]
            ax[0, col].plot(
                t,
                [1000 * (r["truth_x_m"] - r["reference_x_m"]) for r in rows],
                color=color,
                label=mode,
            )
            ax[1, col].plot(
                t, [r["pitch_rad"] for r in rows], color=color, label=mode + " pitch"
            )
            if mode == "Encoder-posture trim":
                ax[1, col].plot(
                    t,
                    [r["pitch_reference_rad"] for r in rows],
                    "--",
                    color="#364050",
                    label="Encoder-derived lean target",
                )
        ax[0, col].set_title(case.replace("-", " "))
        ax[0, col].set_ylabel("Position error (mm)")
        ax[0, col].axhline(25, color="#999", ls=":", lw=1)
        ax[0, col].axhline(-25, color="#999", ls=":", lw=1)
        ax[1, col].set_ylabel("Pitch (rad)")
        ax[1, col].set_xlabel("Physics time (s)")
        for a in ax[:, col]:
            a.grid(alpha=0.2)
            a.legend(fontsize=8)
    path = args.output / f"{name}-trim.png"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
