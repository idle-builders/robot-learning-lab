"""Pair V87 and V88 native contact at the observed specialist boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v88"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    probe_path = OUT / "native-specialist-boundary-probe/matrix.json"
    probe = json.loads(probe_path.read_text())
    if not probe["complete"]:
        raise ValueError("Need complete V88 boundary probe")
    sources = {
        "older": BASE / "v87/native-prior75-retention/matrix.json",
        "routes": BASE / "v87/native-routes63-retention/matrix.json",
    }
    earlier = {}
    for path in sources.values():
        earlier.update({row["id"]: row for row in json.loads(path.read_text())["cases"]})
    rows = []
    for candidate in probe["cases"]:
        old = earlier[candidate["id"]]
        rows.append({
            "id": candidate["id"], "seed": candidate["seed"],
            "v87_contact_s": old["contact_s"],
            "v88_contact_s": candidate["contact_s"],
            "v87_route": old["selection_mode"],
            "v88_route": candidate["selection_mode"],
            "v87_pass": old["pass"], "v88_pass": candidate["pass"],
        })
    fig, ax = plt.subplots(figsize=(9.8, 4.4), layout="constrained")
    x = np.arange(len(rows))
    ax.bar(x - .18, [r["v87_contact_s"] for r in rows], width=.36,
           label="V87", color="#64748b")
    ax.bar(x + .18, [r["v88_contact_s"] for r in rows], width=.36,
           label="V88", color="#0f9f8b")
    ax.axhline(1, color="#dc2626", linestyle="--", linewidth=1.4,
               label="native contact gate · 1 s")
    ax.axhline(1.2, color="#d97706", linestyle=":", linewidth=1.3,
               label="diagnostic margin · 1.2 s")
    ax.set_xticks(x, [str(r["seed"]) for r in rows])
    ax.set_xlabel("Frozen opened start")
    ax.set_ylabel("Uninterrupted fingertip contact · seconds")
    ax.set_title("V88 observed route boundary: paired native reach")
    ax.set_ylim(0, max(5.8, max(r["v88_contact_s"] for r in rows) + .5))
    ax.legend(ncol=2, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=.18)
    fig.savefig(OUT / "specialist-boundary-evaluation.png", dpi=170)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v88-specialist-boundary-summary.v1",
        "role": "opened_paired_native_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v88_probe_matrix_sha256": sha(probe_path),
        "v87_older_matrix_sha256_at_summary": sha(sources["older"]),
        "v87_routes_matrix_sha256": sha(sources["routes"]),
        "case_count": len(rows), "v88_pass_count": sum(r["v88_pass"] for r in rows),
        "minimum_v88_contact_s": min(r["v88_contact_s"] for r in rows),
        "rows": rows,
    }
    (OUT / "specialist-boundary-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"saved V88 paired chart and summary for {len(rows)} native starts")


if __name__ == "__main__":
    main()
