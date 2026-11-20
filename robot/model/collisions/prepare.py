"""Seed R2 from verified R1 geometry/contracts without changing R1 artifacts."""
import shutil
from common import *


def main():
    parent = ROOT / 'artifacts/robot/model/build'
    seal = read(parent / 'evidence-seal.json')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'revision-inputs.json').exists():
        raise ValueError('R2 inputs already prepared; resume collision/build steps without replacing inputs')
    verified = []
    for record in seal['models'] + seal['meshes'] + seal['contracts']:
        path = ROOT / record['path']
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Parent input drift: {path}')
        verified.append(record)
    for record in seal['meshes']:
        source = ROOT / record['path']
        target = OUT / source.relative_to(parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ['frames.json', 'actuators.json', 'sensors.json', 'timing.json',
                 'part-mass-ledger.json', 'mesh-transfer.json', 'collision-model.json',
                 'collision-decomposition.json', 'compliance.json', 'FEM-geometry-identity.json']:
        shutil.copyfile(parent / name, OUT / name)
    shutil.copyfile(parent / 'source-lock.json', OUT / 'parent-source-lock.json')
    write(OUT / 'revision-inputs.json', dict(
        schema='cove50.learning-model.collision-semantics-revision.v1',
        version=VERSION, parent_seal=ref(parent / 'evidence-seal.json'),
        parent_verified_files=len(verified), source_CAD=ref(CAD / 'assembly-parts.json'),
        source_scene=ref(CAD / 'assembly.glb'),
        change='Regenerate affected physical link collision decompositions without source-declared non-BOM unfitted aesthetic-reference shapes.',
        filter=dict(roles=['original_aesthetic_reference', 'ape_shell_reference'],
                    include_in_physical_BOM=False, inspection_reference=True),
        unchanged='Visual geometry, conditional reference-skin mass/inertia allowance, actual manufactured geometry, named22 axes, joints/ranges, sensors, motor runtime and contact parameters.',
        limitation='The retained reference mass distribution is an explicit sensitivity scenario, not measured production skin. Removing an unfitted reference collider does not resolve the physical shell fit or establish walking readiness.',
        source=ref(Path(__file__))))
    print(dict(parent_verified_files=len(verified), copied_native_meshes=len(seal['meshes'])), flush=True)


if __name__ == '__main__':
    main()
