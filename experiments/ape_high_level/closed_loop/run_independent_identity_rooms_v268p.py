"""Frozen new-room identity and owner-response challenge for the route loop.

No scene in this file was used to tune V255/V261/V265 route thresholds.
Geometry is preflighted before actor exposure. The scripted owner response is
withheld until an actor-generated request; evaluator truth only scores claims.
"""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import owner_response_after_no_route_v267p as owner
import run_varied_owner_route_v258p as challenge


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v268p"
VARIATION_SOURCE = Path(challenge.__file__)
OWNER_SOURCE = Path(owner.__file__)
CASES = (
    (186, "present", .30, 9.4, 10.0, 10.4, 12.4),
    (188, "present", .30, 9.8, 10.4, 10.8, 12.8),
    (190, "absent", .30, 9.8, 10.4, 10.8, 12.8),
    (192, "identical_lookalike", .30, 9.2, 9.8, 10.2, 11.8),
    (194, "present", .30, 10.4, 11.0, 11.8, 13.8),
    (196, "identical_lookalike", .30, 9.8, 10.4, 10.8, 12.8),
)
SCENE = {
    186: {"prop_side": 1, "prop_x_m": .02},
    188: {"prop_side": -1, "prop_x_m": -.20},
    190: {"prop_side": 1, "prop_x_m": .02},
    192: {"prop_side": 1, "prop_x_m": .03},
    194: {"prop_side": 1, "prop_x_m": .03},
    196: {"prop_side": -1, "prop_x_m": -.20},
}
OWNER_RESPONSES = {
    186: "show_target", 188: "ignored", 190: "cannot_show",
    192: "show_lookalike", 194: "ignored", 196: "ignored",
}
SENSOR_PROFILE = (
    "virtual head RGB plus 30cm-1m depth, hypothetical near-range "
    "and four-shin first-hit; healthy through all captures")


def main() -> None:
    old = {
        "cases": challenge.CASES,
        "scene": {seed: spec.copy() for seed, spec in challenge.SCENE.items()},
        "out": challenge.OUT,
        "source": challenge.SOURCE,
        "compliant": challenge.COMPLIANT_SEEDS,
        "sensor": challenge.SENSOR_PROFILE,
        "initial": challenge.INITIAL_VETO_HOOK,
        "actor": challenge.ACTOR_POLICY,
        "after": challenge.AFTER_FIRST_VIEW_HOOK,
        "responses": challenge.OWNER_RESPONSES,
        "owner_responses": owner.RESPONSES,
    }
    try:
        challenge.CASES = CASES
        challenge.SCENE.clear()
        challenge.SCENE.update(SCENE)
        challenge.OUT, challenge.SOURCE = OUT, SOURCE
        challenge.COMPLIANT_SEEDS = tuple(seed for seed, response
                                          in OWNER_RESPONSES.items()
                                          if response.startswith("show_"))
        challenge.SENSOR_PROFILE = SENSOR_PROFILE
        challenge.INITIAL_VETO_HOOK = owner.on_initial_veto
        challenge.AFTER_FIRST_VIEW_HOOK = owner.after_first_view
        challenge.ACTOR_POLICY = (
            "V261/V265 active-view, free/occupied/unknown and help policy; "
            "owner response mode is external scene control only after ask")
        challenge.OWNER_RESPONSES = OWNER_RESPONSES
        owner.RESPONSES = OWNER_RESPONSES
        challenge.main()
    finally:
        challenge.CASES = old["cases"]
        challenge.SCENE.clear()
        challenge.SCENE.update(old["scene"])
        challenge.OUT, challenge.SOURCE = old["out"], old["source"]
        challenge.COMPLIANT_SEEDS = old["compliant"]
        challenge.SENSOR_PROFILE = old["sensor"]
        challenge.INITIAL_VETO_HOOK = old["initial"]
        challenge.ACTOR_POLICY = old["actor"]
        challenge.AFTER_FIRST_VIEW_HOOK = old["after"]
        challenge.OWNER_RESPONSES = old["responses"]
        owner.RESPONSES = old["owner_responses"]
    for source in (VARIATION_SOURCE, OWNER_SOURCE):
        (OUT / source.name).write_bytes(source.read_bytes())
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report.update({
        "variation_source_sha256": sha(VARIATION_SOURCE),
        "owner_response_source_sha256": sha(OWNER_SOURCE),
        "challenge_role": "new independent identity/owner-response split",
    })
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
