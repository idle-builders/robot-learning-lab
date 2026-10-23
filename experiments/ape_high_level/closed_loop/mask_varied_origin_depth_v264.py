"""V264: paired short-range head-depth observations for V263 origin views."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_chest_short_depth_v223 import ideal_mask
from audit_head_short_depth_v225 import SHORT
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v264"
V263_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v263"
V263 = V263_ROOT / "varied-room-origin-audit.json"


def main() -> None:
    if not V263.exists():
        raise FileNotFoundError("Run V263 first")
    rows = []
    for original in sorted(V263_ROOT.glob("seed*-light*/*angle-*.npz")):
        target = OUT / original.relative_to(V263_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with np.load(original) as saved:
            payload = {name: saved[name].copy() for name in saved.files}
        ideal = payload["depth_m"]
        masked = ideal_mask(ideal, SHORT)
        payload["depth_m"] = masked
        np.savez_compressed(target, **payload)
        rows.append({"source": str(original.relative_to(V263_ROOT)),
                     "source_sha256": sha(original),
                     "masked_sha256": sha(target),
                     "valid_ideal_pixels": int(np.isfinite(ideal).sum()),
                     "valid_short_pixels": int(np.isfinite(masked).sum())})
    report = {"scope": "V264 copy V263 independent varied-room origin actor frames with only lower-head map depth masked to ideal co-aligned 7-50cm, 84x58deg range/FOV. RGB, pose and timestamps are byte-array equal after reload; ideal simulator depth determines mask, never directly enters later actor map. No real depth holes/noise/extrinsics or physical package.",
              "source_sha256": sha(SOURCE), "v263_sha256": sha(V263),
              "short_range_fov": list(SHORT), "frames": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "varied-short-origin-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("frames", len(rows), "valid short pixels",
          sum(row["valid_short_pixels"] for row in rows), flush=True)


if __name__ == "__main__":
    main()
