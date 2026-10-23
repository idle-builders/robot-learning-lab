"""V334: freeze V333 guard and owner-help sequence on the other screen seed."""

from __future__ import annotations

import json
from pathlib import Path

import run_blocked_owner_help_v317 as owner
import run_live_guard_owner_help_v333 as prior
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v334"
SEED = 172


def main() -> None:
    original = (prior.SEED, prior.CASES, prior.OUT)
    try:
        prior.SEED = SEED
        prior.CASES = tuple(case for case in owner.CASES
                            if case[0] == SEED)
        prior.OUT = OUT
        prior.main()
    finally:
        prior.SEED, prior.CASES, prior.OUT = original
    audit_path = OUT / "live-guard-owner-help-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["scope"] = ("V334 frozen V333 live hazard guard plus scripted owner-help "
                      "present/absent/identical-lookalike sequence on the other "
                      "predeclared screen seed172, paired late moving and parked prop. "
                      "All V333 sensor/action thresholds, post-refusal monitor "
                      "duration and task callbacks are unchanged; only the room "
                      "seed changes. Kinematic motion and scripted owner remain.")
    audit["source_sha256"] = sha(SOURCE)
    audit["v333_source_sha256"] = sha(Path(prior.__file__))
    audit["seed"] = SEED
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
