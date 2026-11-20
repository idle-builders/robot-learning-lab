"""Select the corrected URDF visuals while retaining exact R3 native physics."""
from pathlib import Path
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'artifacts/robot/model/timestep'
OUT = ROOT / 'artifacts/robot/model/urdf'
DATA = ROOT / 'data/urdf-inspection-r1-20261026'
HERE = Path(__file__).parent


def ref(path):
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest)


def main():
    parent = json.loads((SOURCE / 'evidence-seal.json').read_text())
    seal = copy.deepcopy(parent)
    seal['version'] = 'r3-urdf-visual-r1-20261026'
    seal['parent_seal'] = ref(SOURCE / 'evidence-seal.json')
    seal['models'] = [x for x in parent['models'] if not x['path'].endswith('/cove50.urdf')]
    seal['models'] += [ref(OUT / n) for n in ('cove50.urdf', 'cove50-shell-hidden.urdf', 'cove50.ros.urdf', 'cove50-shell-hidden.ros.urdf')]
    seal['meshes'] += [ref(p) for p in sorted((OUT / 'meshes').glob('motor-case-*.obj'))]
    seal['authored_source'] += [ref(p) for p in sorted(HERE.iterdir()) if p.is_file()]
    seal['visual_revision_evidence'] = ref(DATA / 'verification.json')
    seal['scope'] = 'Corrected URDF motor visuals and shell-free view only; parent rigid/compliant MJCF physics, source collision hulls and calibrated/unmeasured contracts unchanged.'
    for group in ('models', 'meshes', 'contracts', 'authored_source'):
        for item in seal[group]:
            if ref(ROOT / item['path'])['sha256'] != item['sha256']:
                raise ValueError('Dependency drift: ' + item['path'])
    seal_path = DATA / 'evidence-seal.json'
    seal_path.write_text(json.dumps(seal, indent=2) + '\n')
    current = json.loads((SOURCE / 'current-model.json').read_text())
    current['version'] = seal['version']
    current['rigid_URDF'] = ref(OUT / 'cove50.urdf')
    current['shell_hidden_URDF'] = ref(OUT / 'cove50-shell-hidden.urdf')
    current['evidence_seal'] = ref(seal_path)
    current['parent_model'] = ref(SOURCE / 'current-model.json')
    current['visual_revision'] = dict(report=ref(DATA / 'verification.json'),
        motor_cases_restored=22, shell_visuals_hidden=45,
        physical_model='Unchanged R3 rigid/compliant MJCF, collisions, mass/inertia, joints, motor/sensor/timing contracts',
        independent_review_directory='data/urdf-inspection-r1-20261026/independent-review')
    current['scope'] = 'Selected R3 physics with restored full CAD motor-case URDF visuals and an optional shell-free URDF. No new mechanical selection, policy or physical calibration.'
    path = ROOT / 'robot/model/current-model.json'
    path.write_text(json.dumps(current, indent=2) + '\n')
    print(json.dumps(dict(pointer=ref(path), seal=ref(seal_path), selected_URDF=current['rigid_URDF'])))


if __name__ == '__main__':
    main()
