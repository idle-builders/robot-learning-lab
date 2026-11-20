"""Locations of this experiment's trained option policies and reports.

The 2026-10-16 pilot models and their training reports are retained evidence in
`artifacts/runs/ape-high-level-task-policy-20261016/` (models/ are offloaded to the
private bucket; records/ stays local). Restore the models before replaying:

    .venv/bin/python -m infra.storage.blob_store restore artifacts/runs/ape-high-level-task-policy-20261016/models

New training writes to the managed evidence folder when run through
`infra/lab storage run NAME -- ...`, otherwise to disposable scratch.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POLICY_RUN = ROOT / "artifacts/runs/ape-high-level-task-policy-20261016"
MODELS = POLICY_RUN / "models"
RECORDS = POLICY_RUN / "records"
OUTPUT = Path(os.environ.get("LAB_EVIDENCE_DIR", ROOT / ".cache/work/task-policy-learning"))


def frozen_policy(recorded: str) -> Path:
    """Resolve a policy path frozen before the models moved out of this folder."""
    path = ROOT / recorded
    return path if path.exists() else MODELS / path.name
