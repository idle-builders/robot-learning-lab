"""Paired repair: ask for help after either recovery gate lacks a route.

Replays V259p's exact predeclared scenes and actor. The only policy change is
V256p's owner-help hook accepting the first blocked continuation stage as a
no-supported-view state, after the same actor-only alternate-route screen.
"""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import run_no_supported_view_owner_help_v256p as help_policy
import run_varied_owner_route_v258p as challenge


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v261p"
BASE = ROOT / "artifacts/ape-articulated-depth-route-v259p"
VARIATION_SOURCE = Path(challenge.__file__)
HELP_SOURCE = Path(help_policy.__file__)


def main() -> None:
    old_scene = {seed: spec.copy() for seed, spec in challenge.SCENE.items()}
    old_out, old_source = challenge.OUT, challenge.SOURCE
    try:
        challenge.SCENE[180]["prop_x_m"] = -.20
        challenge.SCENE[184]["prop_x_m"] = -.20
        challenge.OUT, challenge.SOURCE = OUT, SOURCE
        challenge.main()
    finally:
        challenge.SCENE.clear()
        challenge.SCENE.update(old_scene)
        challenge.OUT, challenge.SOURCE = old_out, old_source
    for source in (VARIATION_SOURCE, HELP_SOURCE):
        (OUT / source.name).write_bytes(source.read_bytes())
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report.update({
        "variation_source_sha256": sha(VARIATION_SOURCE),
        "help_policy_source_sha256": sha(HELP_SOURCE),
        "paired_v259_report_sha256": sha(BASE / "report.json"),
        "repair": "V256p help hook now evaluates the last available blocked "
                  "continuation stage, whether the first or second extra "
                  "view. It still requires no actor-supported 8cm "
                  "alternative before asking the owner.",
    })
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
