"""Plot retained Scout hold-pitch regressions without changing evaluation limits."""

import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    fig.patch.set_facecolor("#f5f3eb")
    for ax, (profile, case) in zip(
        axes, [("lower", "drive-return"), ("nominal", "noisy-biased-imu")], strict=True
    ):
        ax.set_facecolor("#fffef8")
        inputs = []
        for prefix, label, color in [
            ("forearm-accounted", "Before covers", "#607a72"),
            ("shin-covered", "With cover inertia", "#c06c39"),
        ]:
            folder = Path(f"artifacts/scout-{prefix}-{profile}-balance-v1/{case}")
            result_path = folder / "result.json"
            trace_path = folder / "trajectory.jsonl"
            result = json.loads(result_path.read_text())
            inputs.append(result["input"])
            for p in (result_path, trace_path):
                bindings[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            held = [r for r in rows if r["time_s"] > result["input"]["duration_s"] - 2]
            ax.plot(
                [r["time_s"] for r in held],
                [1000 * abs(r["pitch_rad"]) for r in held],
                label=label,
                color=color,
                linewidth=2,
            )
        if inputs[0] != inputs[1]:
            raise ValueError("Changed experiment inputs")
        limit = 1000 * inputs[0]["final_pitch_limit_rad"]
        ax.axhline(
            limit,
            color="#a73b31",
            linestyle="--",
            label=f"Unchanged limit: {limit:g} mrad",
        )
        ax.set_title(f"{profile.title()} mass · {case.replace('-', ' ')}")
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Absolute torso pitch (mrad)")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Scout balance · the final two seconds", fontsize=17)
    fig.savefig(output / "hold-pitch.png", dpi=150)
    plt.close(fig)
    (output / "plotter.py").write_bytes(Path(__file__).read_bytes())
    report = dict(
        status="pass",
        scope="Plot of simulated truth; no physical or continuous contact qualification",
        source_bindings=bindings,
        output_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
