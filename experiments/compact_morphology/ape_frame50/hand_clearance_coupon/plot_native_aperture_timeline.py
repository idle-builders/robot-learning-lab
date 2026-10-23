"""Plot retained native contact and cube drift across the two closure diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-hand-clearance-coupon"
OUT = BASE / "native-aperture-comparison-v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for version in ("v3", "v4", "v5", "v6"):
        path = BASE / f"native-aperture-entry-{version}/development/seed-2906.json"
        report = json.loads(path.read_text())
        cube0 = np.asarray(report["route_cube_xyz_m"][:2])
        rows = [r for r in report["timeseries"] if r["time_s"] >= 9.5]
        runs[version] = {
            "time": np.asarray([r["time_s"] for r in rows]),
            "both": np.asarray([r["finger_contact"] and r["palm_contact"] for r in rows]),
            "palm": np.asarray([r["palm_contact"] for r in rows]),
            "finger": np.asarray([r["finger_contact"] for r in rows]),
            "drift": np.asarray([np.linalg.norm(np.asarray(r["cube_xyz_m"][:2]) - cube0) * 1000 for r in rows]),
            "preclose_xy": np.asarray((report.get("closure_source_check") or
                                       {"cube_xyz_m": next(r["cube_xyz_m"] for r in rows
                                                           if r["time_s"] >= 12.4)})["cube_xyz_m"][:2]),
            "cube_xy": np.asarray([r["cube_xyz_m"][:2] for r in rows]),
            "source": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    fig, (contact, drift) = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True,
                                       gridspec_kw={"height_ratios": [1, 1.5]}, layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    colors = {"v3": "#2563eb", "v4": "#d97706"}
    labels = {"v3": "Early entry + fixed-arm close", "v4": "Early entry + coupled arm/jaw close"}
    for version, lane in (("v3", 1), ("v4", 0)):
        run = runs[version]
        contact.fill_between(run["time"], lane - .31, lane + .31,
                             where=run["both"], step="post", color=colors[version], alpha=.9)
        contact.plot([], [], color=colors[version], linewidth=6, label=labels[version])
        drift.plot(run["time"], run["drift"], color=colors[version], linewidth=2.2,
                   label=labels[version])
    for ax in (contact, drift):
        ax.axvspan(11.0, 12.4, color="#16a34a", alpha=.08, zorder=-1)
        ax.axvline(12.4, color="#334155", linestyle="--", linewidth=1)
        ax.grid(axis="x", color="#dbe2ea", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    contact.set_yticks([0, 1], ["Coupled v4", "Fixed-arm v3"])
    contact.set_ylim(-.55, 1.55)
    contact.set_ylabel("Both jaw pads touching")
    contact.set_title("Cove-50 aperture entry: open contact does not survive powered closure",
                      loc="left", fontsize=13, weight="bold")
    contact.text(11.7, 1.46, "open-jaw hold", ha="center", color="#166534", fontsize=9)
    contact.text(12.45, 1.46, "close begins", ha="left", color="#334155", fontsize=9)
    drift.axhline(5, color="#dc2626", linestyle=":", linewidth=1.5, label="5 mm gate")
    drift.set_ylabel("Cube XY drift from entry start (mm)")
    drift.set_xlabel("Time in continuous native episode (s)")
    drift.set_xlim(9.5, 14.4)
    drift.set_ylim(0, 7)
    drift.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    fig.savefig(OUT / "contact-drift-timeline.png", dpi=170)
    plt.close(fig)

    fig, (contact, drift) = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True,
                                       gridspec_kw={"height_ratios": [1.2, 1.5]}, layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    lanes = (("v5", "palm", 3, "#7c3aed"), ("v5", "finger", 2, "#a78bfa"),
             ("v6", "palm", 1, "#b91c1c"), ("v6", "finger", 0, "#f87171"))
    for version, pad, lane, color in lanes:
        run = runs[version]
        contact.fill_between(run["time"], lane - .31, lane + .31,
                             where=run[pad], step="post", color=color, alpha=.9)
    for ax in (contact, drift):
        ax.axvline(12.4, color="#334155", linestyle="--", linewidth=1)
        ax.axvspan(14.4, 15.4, color="#16a34a", alpha=.07, zorder=-1)
        ax.grid(axis="x", color="#dbe2ea", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    contact.set_yticks([0, 1, 2, 3], ["v6 moving pad", "v6 fixed pad",
                                       "v5 moving pad", "v5 fixed pad"])
    contact.set_ylim(-.55, 3.55)
    contact.set_ylabel("Individual pad contact")
    contact.set_title("Static contact path passes; powered position control loses both pads",
                      loc="left", fontsize=13, weight="bold")
    contact.text(14.9, 3.45, "hold", ha="center", color="#166534", fontsize=9)
    for version, color, label in (("v5", "#7c3aed", "Open-loop contact path"),
                                  ("v6", "#b91c1c", "Simulator-truth pad IK")):
        run = runs[version]
        from_close = run["time"] >= 12.4
        drift.plot(run["time"][from_close],
                   np.linalg.norm(run["cube_xy"][from_close] - run["preclose_xy"], axis=1) * 1000,
                   color=color, linewidth=2.2, label=label)
    drift.axhline(2, color="#dc2626", linestyle=":", linewidth=1.5,
                  label="v6 preclosure drift gate")
    drift.set_ylabel("Cube XY drift from preclosure (mm)")
    drift.set_xlabel("Time in continuous native episode (s)")
    drift.set_xlim(12.4, 15.4)
    drift.set_ylim(0, 4.5)
    drift.legend(loc="upper right", frameon=False, fontsize=8)
    fig.savefig(OUT / "contact-feedback-timeline.png", dpi=170)
    plt.close(fig)
    (OUT / "provenance.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-aperture-timeline.v1",
        "purpose": "Display-only plot from retained native 50 Hz reports; no physics or gate recomputation.",
        "source_reports": {key: {"path": value["source"], "sha256": value["sha256"]}
                           for key, value in runs.items()},
        "plot": "artifacts/ape-cove50-hand-clearance-coupon/native-aperture-comparison-v1/contact-drift-timeline.png",
        "feedback_plot": "artifacts/ape-cove50-hand-clearance-coupon/native-aperture-comparison-v1/contact-feedback-timeline.png",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
