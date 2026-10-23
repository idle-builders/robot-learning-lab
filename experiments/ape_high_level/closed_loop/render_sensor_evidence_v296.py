"""V296: human-readable Direct Frame sensor and stop-tail evidence figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v296"
V291 = ROOT / "artifacts/ape-closed-loop-cove50-v291/varied-low-clutter-audit.json"
V293 = ROOT / "artifacts/ape-closed-loop-cove50-v293/wider-head-sweep-audit.json"
V294 = ROOT / "artifacts/ape-closed-loop-cove50-v294/heldout-head-sweep-audit.json"
V295 = ROOT / "artifacts/ape-closed-loop-cove50-v295/low-clutter-stop-tail-audit.json"


def chart(dev: dict, wider: dict, fresh: dict, stop: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": (1., 1.25)})
    labels = ("shin 4\nrooms 139–148", "head ±1.5\nrooms 139–148",
              "shin 4\nrooms 149–158", "head ±1.5\nrooms 149–158")
    source_rows = (dev["scores"]["shin_four"]["0.065"],
                   wider["scores"]["head_1p5"],
                   fresh["scores"]["shin_four"],
                   fresh["scores"]["head_1p5"])
    unsafe = [row["false_safe"] for row in source_rows]
    blocked = [row["false_block"] for row in source_rows]
    x = np.arange(4)
    axes[0].bar(x - .18, unsafe, .36, color="#b94e4a", label="unsafe passed")
    axes[0].bar(x + .18, blocked, .36, color="#cfac4b", label="safe blocked")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 2.25)
    axes[0].set_yticks((0, 1, 2))
    axes[0].set_ylabel("Action count")
    axes[0].set_title("Same-action point-gap decisions")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].set_axisbelow(True)
    colors = {(.04, "executed_shin_search"): "#446a90",
              (.08, "executed_shin_search"): "#ae684a",
              (.04, "counterfactual_second_step"): "#8c568e"}
    names = {(.04, "executed_shin_search"): "4 cm, executed route",
             (.08, "executed_shin_search"): "8 cm, executed route",
             (.04, "counterfactual_second_step"): "4 cm, deeper hypothetical"}
    for row in stop["rows"]:
        key = row["physical_object_height_m"], row["route_state"]
        curve = row["rows"]
        axes[1].plot([item["extra_m"] * 100 for item in curve],
                     [item["model_clearance_m"] * 100 for item in curve],
                     color=colors[key], linewidth=2.2, label=names[key])
    axes[1].axhline(5, color="#b94e4a", linestyle="--",
                    linewidth=1.5, label="5 cm target")
    axes[1].axhline(0, color="#4a5158", linewidth=.8)
    axes[1].set_xlim(0, stop["source_upper_path_to_observed_stop_m"] * 100)
    axes[1].set_xlabel("Hypothetical extra straight travel (cm)")
    axes[1].set_ylabel("Selected-body modeled clearance (cm)")
    axes[1].set_title("Source-body stop-tail stress only")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=.2)
    fig.suptitle("Cove-50 Direct Frame: low-object observation and stopping gap",
                 fontsize=15)
    fig.tight_layout()
    path = OUT / "sensor-and-stop-comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def rooms(dev: dict, fresh: dict) -> Path:
    cases = [(V291.parent, case) for case in dev["preflight"]] + [
        (V294.parent, case) for case in fresh["preflight"]]
    tile_w, tile_h = 260, 213
    sheet = Image.new("RGB", (tile_w * 5, tile_h * 4), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (root, case) in enumerate(cases):
        path = root / f"seed{case['seed']}-{case['kind']}.png"
        photo = Image.open(path).convert("RGB")
        photo.thumbnail((tile_w, tile_h - 28))
        col, row = index % 5, index // 5
        x, y = col * tile_w, row * tile_h
        sheet.paste(photo, (x, y + 28))
        draw.rectangle((x, y, x + tile_w, y + 27), fill="#263746")
        draw.text((x + 5, y + 6),
                  f"{case['seed']} {case['kind']} | floor object {case['crate_full_size_xyz_m'][2]*100:.0f} cm",
                  fill="white")
    path = OUT / "twenty-room-contact-sheet.png"
    sheet.save(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev, wider, fresh, stop = (json.loads(path.read_text())
                               for path in (V291, V293, V294, V295))
    comparison = chart(dev, wider, fresh, stop)
    contact_sheet = rooms(dev, fresh)
    report = {"scope": "V296 display-only figures from frozen V291/V293/V294 point-gap screens and V295 evaluator-only source-body stop-tail stress. The chart is a summary, not a new experiment or selected-frame safety claim; the contact sheet shows the 20 actual rendered room variants.",
              "source_sha256": sha(SOURCE),
              "matplotlib_version": matplotlib.__version__,
              "v291_sha256": sha(V291), "v293_sha256": sha(V293),
              "v294_sha256": sha(V294),
              "v295_sha256": sha(V295),
              "comparison_png_sha256": sha(comparison),
              "room_sheet_png_sha256": sha(contact_sheet)}
    (OUT / "figure-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
