"""Presentation-only scene for native walking replays; no simulation changes."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def reference(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.relative_to(ROOT)), sha256=digest, bytes=path.stat().st_size)


def prepare(model_path, selected):
    """Replace only non-contact motor visuals with the selected repaired meshes."""
    source = mujoco.MjModel.from_xml_path(str(model_path))
    tree = ET.parse(model_path)
    root = tree.getroot()
    assets = root.find('asset')
    meshes = {m.get('name'): m for m in assets.findall('mesh')}
    geoms = {g.get('name'): g for g in root.findall('.//geom')}
    for mesh in meshes.values():
        mesh.set('file', str(model_path.parent / mesh.get('file')))
    urdf_ref = selected['rigid_URDF']
    urdf = ROOT / urdf_ref['path']
    if reference(urdf)['sha256'] != urdf_ref['sha256']:
        raise ValueError('Selected repaired URDF identity changed')
    repaired = []
    for link in ET.parse(urdf).getroot().findall('link'):
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None or not mesh.get('filename', '').startswith('meshes/motor-case-'):
                continue
            name = visual.get('name')
            geom = geoms[name]
            if geom.get('contype') != '0' or geom.get('conaffinity') != '0':
                raise ValueError('Motor visual unexpectedly participates in contact')
            # These selected exports store vertices directly in the joint link
            # frame. Refuse a future transformed visual instead of misplacing it.
            origin = visual.find('origin')
            if origin is not None or geom.get('pos', '0 0 0') != '0 0 0':
                raise ValueError('Motor visual transform needs an explicit mapping')
            native_mesh = meshes[geom.get('mesh')]
            previous = native_mesh.get('file')
            path = urdf.parent / mesh.get('filename')
            native_mesh.set('file', str(path))
            geom.set('rgba', '.10 .35 .68 1')
            repaired.append(dict(visual=name, link=link.get('name'),
                original_mesh=str(Path(previous).relative_to(ROOT)), mesh=reference(path)))
    if len(repaired) != 22:
        raise ValueError('Selected complete motor housing inventory differs')
    ET.SubElement(assets, 'texture', dict(name='replay-grey-sky', type='skybox',
        builtin='gradient', rgb1='.80 .82 .84', rgb2='.80 .82 .84', width='512', height='3072'))
    geoms['floor'].set('rgba', '.80 .81 .82 1')
    for light in root.findall('.//light'):
        light.set('castshadow', 'false')
        light.set('directional', 'true')
        light.set('diffuse', '.40 .40 .40')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    # The returned model is used only with mj_forward and rendering. Also record
    # exact physical identity so a visual repair cannot silently change the body.
    fields = ('body_parentid', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
        'body_pos', 'body_quat', 'jnt_type', 'jnt_bodyid', 'jnt_pos', 'jnt_axis',
        'jnt_range', 'jnt_limited', 'dof_damping', 'dof_armature', 'dof_frictionloss',
        'actuator_gear', 'actuator_ctrlrange', 'actuator_forcerange', 'qpos0',
        'geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_solref', 'geom_solimp')
    for field in fields:
        if not np.array_equal(getattr(source, field), getattr(model, field)):
            raise ValueError('Render copy changed physical field: '+field)
    contact = (source.geom_contype != 0) | (source.geom_conaffinity != 0)
    for field in ('geom_type', 'geom_dataid', 'geom_bodyid', 'geom_size', 'geom_pos', 'geom_quat'):
        if not np.array_equal(getattr(source, field)[contact], getattr(model, field)[contact]):
            raise ValueError('Render copy changed contact geometry: '+field)
    model.vis.headlight.ambient[:] = [.55, .55, .55]
    model.vis.headlight.diffuse[:] = [.50, .50, .50]
    model.vis.headlight.specular[:] = [.12, .12, .12]
    proof = dict(schema='cove50.walk-replay-visuals.v1', native_model=reference(model_path),
        visual_urdf=urdf_ref, repaired_motor_housings=repaired,
        physical_fields_exact=list(fields), contact_geometry_exact=True,
        background='light grey sky and floor',
        motor_color='blue display highlight; source geometry unchanged',
        helper=reference(Path(__file__)), no_new_physics_rollout=True)
    return model, proof
