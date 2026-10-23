"""Verify the retained local CAS-backed robot before a control experiment."""
import hashlib
import json
from pathlib import Path, PurePosixPath

EXPECTED_BUNDLE='0db025c67e87e1b93ea6027d107cda9450a29eb9ae9991b59d4b462ff5315db4'


def verify_control_bundle(bundle):
    bundle=Path(bundle)
    raw=(bundle/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_BUNDLE:raise ValueError('Unexpected Pip bundle manifest')
    manifest=json.loads(raw);size=0
    for name,expected in manifest['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Nonportable bundle path')
        data=(bundle/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Pip source integrity failure: '+name)
        size+=len(data)
    return {'status':'pass','manifest_sha256':EXPECTED_BUNDLE,'files':len(manifest['files']),'bytes':size,
        'materialization':'read existing local CAS links; no CAD copy'}
