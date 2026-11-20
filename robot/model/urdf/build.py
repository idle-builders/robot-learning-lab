"""Export and render the selected URDF without appearance shells; no dynamics."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'artifacts/robot/model/timestep'
HERE = Path(__file__).parent
OUT = ROOT / 'artifacts/robot/model/urdf'
DATA = ROOT / 'data/urdf-inspection-r1-20261026'


def ref(p):
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p.relative_to(ROOT)), sha256=digest, bytes=p.stat().st_size)


def package():
    """Repack active generated inputs without rerendering or changing meshes."""
    for name in ('README.md', 'source-model.json'):
        shutil.copyfile(HERE / name, OUT / name)
    bundle = OUT / 'cove50-urdf-shell-hidden.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p != bundle and p.suffix != '.png':
                item = zipfile.ZipInfo(str(p.relative_to(OUT)), date_time=(2026, 10, 26, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = 0o100644 << 16
                z.writestr(item, p.read_bytes())
    return bundle


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    original = ET.parse(SOURCE / 'cove50.urdf')
    shutil.copyfile(SOURCE / 'cove50.urdf', HERE / 'cove50-source.urdf')
    # Restore complete CAD motor surfaces rather than the old 4,000-face
    # simplifications. Contact hulls and all explicit inertials remain exact.
    scene_path = ROOT / 'artifacts/cove50-electronics-integration/r1-20261026/assembly.glb'
    source_meta = json.loads((SOURCE / 'mesh-transfer.json').read_text())
    if ref(scene_path)['sha256'] != source_meta['source_scene']['sha256']:
        raise ValueError('CAD scene identity changed')
    scene = trimesh.load(scene_path, force='scene')
    manifest = json.loads((scene_path.parent / 'assembly-parts.json').read_text())
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    origins = {'torso': rotation @ np.array([0., 0., .300])}
    origins.update({j['child']: rotation @ (np.array(j['origin_world_mm']) / 1000) for j in manifest['joints']})
    motor_meshes = []
    for link in original.getroot().findall('link'):
        for visual in link.findall('visual'):
            name = visual.get('name', '')
            if 'case' not in name or not any(s in name for s in ('C047', 'XM335', 'servo')):
                continue
            transform, key = scene.graph[name]
            mesh = scene.geometry[key].copy()
            mesh.apply_transform(transform)
            mesh.vertices = mesh.vertices @ rotation.T - origins[link.get('name')]
            filename = f'meshes/motor-case-{len(motor_meshes):02d}.obj'
            dest = OUT / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_name('.' + dest.stem + '.tmp.obj')
            mesh.export(temporary)
            temporary.replace(dest)
            visual.find('geometry/mesh').set('filename', filename)
            motor_meshes.append(dict(name=name, faces=len(mesh.faces), mesh=ref(dest)))
    original.write(HERE / 'cove50.urdf', encoding='utf-8', xml_declaration=True)
    bare = copy.deepcopy(original)
    removed = []
    motors = []
    for link in bare.getroot().findall('link'):
        for visual in list(link.findall('visual')):
            name = visual.get('name', '')
            if name.startswith('ape_body_shell__') or name == 'original_profile_head_skin_v23':
                removed.append(name)
                link.remove(visual)
            elif 'case' in name and any(s in name for s in ('C047', 'XM335', 'servo')):
                motors.append(dict(link=link.get('name'), visual=name))
    for tag in ('inertial', 'collision', 'joint'):
        before = [ET.tostring(x) for x in original.findall('.//' + tag)]
        after = [ET.tostring(x) for x in bare.findall('.//' + tag)]
        if before != after:
            raise ValueError(f'Physical URDF {tag} changed')
    if len(motors) != 22 or len(removed) != 45:
        raise ValueError('Motor or shell inventory differs from selected model')
    bare.write(HERE / 'cove50-shell-hidden.urdf', encoding='utf-8', xml_declaration=True)
    for tree, name in ((original, 'cove50.ros.urdf'), (bare, 'cove50-shell-hidden.ros.urdf')):
        ros = copy.deepcopy(tree)
        for extension in list(ros.getroot().findall('mujoco')):
            ros.getroot().remove(extension)
        for mesh in ros.findall('.//mesh'):
            mesh.set('filename', 'package://cove50_description/' + mesh.get('filename'))
        ros.write(HERE / name, encoding='utf-8', xml_declaration=True)
    # Materialize the package so standard relative mesh paths work unchanged.
    for name in ('cove50.urdf', 'cove50-shell-hidden.urdf', 'cove50.srdf',
                 'actuators.json', 'sensors.json', 'timing.json', 'frames.json',
                 'link-inertias.json', 'source-model.json', 'compliance.json',
                 'compliance-mass-splits.json', 'part-mass-ledger.json', 'source-lock.json',
                 'cove50.ros.urdf', 'cove50-shell-hidden.ros.urdf', 'package.xml',
                 'CMakeLists.txt', 'README.md'):
        shutil.copyfile(HERE / name, OUT / name)
    mesh_refs = sorted({m.get('filename') for m in original.findall('.//mesh')})
    for name in mesh_refs:
        dest = OUT / name
        if name.startswith('meshes/motor-case-'):
            continue
        src = SOURCE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dest)
    for p in (OUT / 'meshes').iterdir():
        if p.is_file() and str(p.relative_to(OUT)) not in mesh_refs:
            p.unlink()
    model = mujoco.MjModel.from_xml_path(str(OUT / 'cove50-shell-hidden.urdf'))
    full = mujoco.MjModel.from_xml_path(str(OUT / 'cove50.urdf'))
    selected = mujoco.MjModel.from_xml_path(str(SOURCE / 'cove50.urdf'))
    for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_pos', 'jnt_axis', 'jnt_range'):
        if not np.array_equal(getattr(full, name), getattr(model, name)):
            raise ValueError(f'Compiled physical field changed: {name}')
        if not np.array_equal(getattr(selected, name), getattr(model, name)):
            raise ValueError(f'Selected physical field changed: {name}')
    motor_names = {x['visual'] for x in motors}
    found = []
    # Color only the render copy to make the actual motor-case meshes legible.
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name in motor_names:
            model.geom_rgba[i] = [.08, .38, .85, 1.]
            model.geom_matid[i] = -1
            found.append(name)
    if set(found) != motor_names:
        raise ValueError('Motor visual missing from URDF import')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 0, 0, 0, 0]
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 1000
    model.vis.headlight.ambient[:] = [.65, .65, .65]
    model.vis.headlight.diffuse[:] = [.8, .8, .8]
    renderer = mujoco.Renderer(model, height=1000, width=800)
    canvas = Image.new('RGB', (1600, 1080), '#121b25')
    camera = mujoco.MjvCamera()
    camera.distance = .88
    camera.elevation = -10
    camera.lookat[:] = [0, 0, -.035]
    for column, azimuth in enumerate((40, 220)):
        camera.azimuth = azimuth
        renderer.update_scene(data, camera=camera, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        canvas.paste(Image.fromarray(renderer.render()), (800 * column, 80))
    renderer.close()
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 15), 'SELECTED COVE-50 URDF | SHELL HIDDEN | FRONT / REAR', fill='white', font_size=25)
    draw.text((30, 49), '22 motor cases shown in blue (display only) | mass 3.8303 kg unchanged | neutral joint pose', fill='#b6cee8', font_size=20)
    canvas.save(OUT / 'cove50-shell-hidden.png')
    bundle = package()
    report = dict(schema='cove50.urdf-shell-hidden-inspection.v1', source=ref(SOURCE / 'cove50.urdf'),
        bare_urdf=ref(HERE / 'cove50-shell-hidden.urdf'), mass_kg=float(model.body_mass.sum()),
        active_joints=sum(j.get('type') not in ('fixed', 'floating') for j in original.findall('joint')),
        source_links=len(original.findall('link')), mesh_files=len(mesh_refs), missing_meshes=[],
        motor_cases=motors, motor_visual_count=len(found), removed_visuals=removed,
        full_resolution_motor_meshes=motor_meshes, source_scene=ref(scene_path),
        unchanged=['all inertials and mass', 'all collision geometry', 'all joints, axes and travel', 'actuator and sensor sidecars'],
        picture=ref(OUT / 'cove50-shell-hidden.png'), portable_bundle=ref(bundle),
        build_recipe=ref(Path(__file__)),
        regeneration='.venv-control/bin/python robot/model/urdf/build.py',
        scope='Static rendering of the actual compiled URDF. Motor blue is render-only. No physics steps or learning. Importing URDF alone in MuJoCo fixes the base and does not apply SRDF; learning needs the released floating-base MJCF and motor/timing contracts.',
        remaining_assumptions='Physical hip travel, servo calibration and compliance remain as recorded in R3. This export does not qualify walking or hardware.')
    (DATA / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('mass_kg', 'active_joints', 'source_links', 'mesh_files', 'motor_visual_count', 'picture', 'portable_bundle')}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-only', action='store_true')
    args = parser.parse_args()
    if args.package_only:
        report = json.loads((DATA / 'verification.json').read_text())
        report['portable_bundle'] = ref(package())
        report['build_recipe'] = ref(Path(__file__))
        (DATA / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['portable_bundle']))
    else:
        main()
