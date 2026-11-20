"""Reuse exact pinned source/CUDA attestation, no task/model construction."""
from common import project
exec(compile(project('control/skills/forward-stop-learning/compute/attest.py',
    '6ded3391b5931696ce92b9df11aec3ca1874d0c1aa3c36c4fc6e233fb152595a', []),
    __file__ + ':retained-attestation', 'exec'), globals())
