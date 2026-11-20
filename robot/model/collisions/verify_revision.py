"""Bind the collision-only revision and verify unchanged native body contracts."""
import importlib.metadata
import platform
import sys
import xml.etree.ElementTree as ET
import mujoco
from common import *


def main():
    parent = ROOT / 'artifacts/robot/model/build'
    records = []
    for name in ['cove50.urdf', 'cove50-rigid.xml', 'cove50-compliant.xml']:
        old = mujoco.MjModel.from_xml_path(str(parent / name))
        new = mujoco.MjModel.from_xml_path(str(OUT / name))
        checked = {}
        for attr in ['nq', 'nv', 'nu', 'nbody', 'njnt', 'nsensor']:
            checked[attr] = int(getattr(new, attr))
            if getattr(new, attr) != getattr(old, attr):
                raise ValueError(f'Unexpected native dimension change: {name}/{attr}')
        unchanged_arrays = ['body_mass', 'body_ipos', 'body_inertia', 'body_iquat',
                            'body_parentid', 'body_pos', 'body_quat',
                            'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited',
                            'dof_damping', 'dof_armature', 'actuator_trnid', 'actuator_gear',
                            'actuator_ctrlrange', 'site_pos', 'site_quat']
        for attr in unchanged_arrays:
            if not np.array_equal(getattr(new, attr), getattr(old, attr)):
                raise ValueError(f'Unexpected body contract change: {name}/{attr}')
        od, nd = mujoco.MjData(old), mujoco.MjData(new)
        mujoco.mj_forward(old, od); mujoco.mj_forward(new, nd)
        for attr in ['xpos', 'xmat', 'xipos', 'site_xpos', 'site_xmat']:
            if not np.array_equal(getattr(nd, attr), getattr(od, attr)):
                raise ValueError(f'Unexpected neutral FK change: {name}/{attr}')
        records.append(dict(model=ref(OUT / name), native=checked,
                            total_mass_kg=float(new.body_mass.sum()),
                            unchanged_arrays=unchanged_arrays,
                            neutral_FK_max_error_m=float(np.max(abs(nd.xpos - od.xpos))),
                            active_geoms=int(np.count_nonzero(new.geom_contype | new.geom_conaffinity))))
    decomp = read(OUT / 'collision-decomposition.json')
    manifest = {p['name']: p for p in read(CAD / 'assembly-parts.json')['parts']}
    expected = {name for name, p in manifest.items()
                if p.get('role') in ['original_aesthetic_reference', 'ape_shell_reference']
                and p.get('include_in_physical_BOM') is False
                and p.get('inspection_reference') is True}
    if set(decomp['excluded_unfitted_aesthetic_references']) != expected:
        raise ValueError('Reference classification identity drift')
    if any(expected.intersection(link['source_parts']) for link in decomp['links'].values()):
        raise ValueError('Reference remains in physical collision inputs')
    models = [ref(OUT / name) for name in ['cove50.urdf', 'cove50.ros.urdf', 'cove50-rigid.xml', 'cove50-compliant.xml']]
    meshpaths = set()
    for record in models:
        for node in ET.parse(ROOT / record['path']).iter('mesh'):
            filename = node.get('file', node.get('filename', '')).replace('package://cove50_description/', '')
            if filename:
                meshpaths.add(OUT / filename)
    mass = read(OUT / 'link-inertias.json')
    validation = dict(schema='cove50.collision-revision-validation.v1', version=VERSION,
                      parent_seal=ref(parent / 'evidence-seal.json'), models=records,
                      excluded_reference_parts=sorted(expected), mesh_dependencies=len(meshpaths),
                      conditional_reference_skin_mass_kg=mass['reference_skin_mass_kg'],
                      observation='The source-declared unfitted reference shapes remain visible and carry the same explicit mass/inertia sensitivity allowance, but no longer generate rigid collision forces. Physical geometry, joint tree, actuator/sensor contracts and neutral FK are unchanged.',
                      limitations=['No native dynamics or learned gait is established by compilation/FK identity.',
                                   'V-HACD hulls remain approximate, including real-part gap filling and thin-wall error.',
                                   'Unfitted body panels still require design/clearance work; this model correction does not fit them physically.',
                                   'Inherited adjacent two-hop contact exclusions can hide collisions; no new exclusion is added.'])
    write(OUT / 'native-revision-validation.json', validation)
    sources = [ref(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    write(OUT / 'source-lock.json', dict(schema='cove50.collision-revision-source-lock.v1', version=VERSION,
          parent_source_lock=ref(parent / 'source-lock.json'), inputs=ref(OUT / 'revision-inputs.json'),
          authored_sources=sources, model=models, validation=ref(OUT / 'native-revision-validation.json')))
    contracts = [ref(OUT / name) for name in ['source-lock.json', 'frames.json', 'actuators.json', 'sensors.json',
                 'timing.json', 'part-mass-ledger.json', 'link-inertias.json', 'compliance.json',
                 'collision-decomposition.json', 'native-revision-validation.json']]
    seal = dict(schema='cove50.learning-model.seal.v1', version=VERSION,
                parent_seal=ref(parent / 'evidence-seal.json'), authored_source=sources,
                models=models, meshes=[ref(p) for p in sorted(meshpaths)], contracts=contracts,
                environment=dict(python=sys.version, platform=platform.platform(), packages={n: importlib.metadata.version(n) for n in ['mujoco', 'numpy', 'scipy', 'trimesh', 'vhacdx']}),
                qualification='Exact collision-only derivative identities and native compilation/FK observations. Dynamics, actual fitted shell and learned behavior require their own evidence.')
    write(OUT / 'evidence-seal.json', seal)
    pointer = read(parent / 'current-model.json')
    pointer.update(version=VERSION, status='provisional_reference_collision_corrected_learning_model',
                   rigid_URDF=ref(OUT / 'cove50.urdf'), rigid_MJCF=ref(OUT / 'cove50-rigid.xml'),
                   compliant_MJCF=ref(OUT / 'cove50-compliant.xml'),
                   actuators=ref(OUT / 'actuators.json'), sensors=ref(OUT / 'sensors.json'),
                   report=ref(OUT / 'native-revision-validation.json'),
                   parent_model=ref(parent / 'current-model.json'), evidence_seal=ref(OUT / 'evidence-seal.json'),
                   scope='Collision-corrected derivative for matched native diagnosis; no physical shell fit, stable walking, or tuning readiness is implied. CAD selection unchanged.')
    write(OUT / 'current-model.json', pointer)
    print(dict(compiled_native_models=len(records), excluded_reference_parts=len(expected),
               mesh_dependencies=len(meshpaths), estimated_mass_kg=mass['total_mass_kg']), flush=True)


if __name__ == '__main__':
    main()
