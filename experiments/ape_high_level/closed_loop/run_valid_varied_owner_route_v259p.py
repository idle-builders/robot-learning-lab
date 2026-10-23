"""Geometry-valid opposite-side challenge, frozen before actor exposure.

V258p's first declaration failed evaluator-only geometry preflight before any
owner episode: the mirrored prop intersected a crate. A geometry-only x sweep
found -20 cm clear of the crate for both mirrored cases. No actor result or
item image informed this replacement declaration.
"""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import run_varied_owner_route_v258p as challenge


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v259p"
VARIATION_SOURCE = Path(challenge.__file__)


def main() -> None:
    old_scene = {seed: spec.copy() for seed, spec in challenge.SCENE.items()}
    old_out, old_source = challenge.OUT, challenge.SOURCE
    try:
        challenge.SCENE[180]["prop_x_m"] = -.20
        challenge.SCENE[184]["prop_x_m"] = -.20
        challenge.OUT = OUT
        challenge.SOURCE = SOURCE
        challenge.main()
    finally:
        challenge.SCENE.clear()
        challenge.SCENE.update(old_scene)
        challenge.OUT, challenge.SOURCE = old_out, old_source
    (OUT / VARIATION_SOURCE.name).write_bytes(VARIATION_SOURCE.read_bytes())
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["variation_source_sha256"] = sha(VARIATION_SOURCE)
    report["geometry_revision_reason"] = (
        "V258p's predeclared mirrored x=-2/-3cm path intersected the crate "
        "before actor exposure. Evaluator-only x sweep found x=-20cm "
        "geometry-valid for both mirrored cases; no actor result guided "
        "this replacement declaration.")
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
