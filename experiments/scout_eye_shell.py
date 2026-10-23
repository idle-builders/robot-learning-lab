"""Bounded rear-open eye study using the shared CAD recipe and native exports."""
import argparse
import json
import math
from pathlib import Path
import traceback

from build123d import Compound, Pos, Rot, export_step, export_stl, import_step
import yaml

from lab.config import load_config
from lab.face_layout import check_face
from lab.mechanical import decorative_shape, part_shape, primitive
from lab.robot import make_robot
from tools.check_camera_support import copy_source, sha, write


def volume(shape):
    return sum(s.volume for s in shape.solids())


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'fail', 'physical_status': 'not_run', 'installed_mass_status': 'not_run',
              'scope': 'Scout eye-shell geometry and conditional material delta; not complete head qualification'}
    files = ['experiments/scout_eye_shell.py', 'lab/config.py', 'lab/mechanical.py',
             'lab/styling.py', 'lab/face_layout.py', 'lab/robot.py', 'robot/base.yaml',
             'variants/scout-shell-eyes.yaml', 'variants/scout-distal-grip.yaml',
             'robot/printing-cost.yaml']
    report['source_hashes'] = copy_source(output, {p: Path(p) for p in files})
    try:
        assumptions = yaml.safe_load(Path('robot/printing-cost.yaml').read_text())
        cfg = load_config(Path('variants/scout-shell-eyes.yaml'))
        original_robot = make_robot(load_config(Path('variants/scout-distal-grip.yaml')))
        meshes, rows = [], []
        for wall in (None, .0012, .002, .003):
            cfg.face.eye_shroud_wall_m = wall
            robot = make_robot(cfg)
            # Name is the only expected interface difference between variants.
            assert robot.joints == original_robot.joints
            assert robot.links == original_robot.links
            name = 'solid' if wall is None else f'wall-{wall*1000:g}mm'
            folder = output / name
            folder.mkdir()
            write(folder / 'resolved-config.json', cfg.model_dump())
            face_check = check_face(robot)
            head = next(l for l in robot.links if l.name == 'head')
            eye_parts = [(i,g) for i,g in enumerate(head.geometry) if g.role.startswith('eye_')]
            both = []
            for index, g in eye_parts:
                shape = part_shape(robot, head, index)
                placed = Pos(*(v*1000 for v in g.pos)) * Rot(*(math.degrees(v) for v in g.rpy)) * shape
                key = f'{g.role}-{index}'
                path = folder / f'{key}.step'
                export_step(placed, path)
                restored = import_step(path)
                assert restored.is_valid and len(restored.solids()) == 1
                # Native STEP readback must preserve all shape stock, not just volume.
                assert volume(placed.cut(restored)) < .0001
                assert volume(restored.cut(placed)) < .0001
                export_stl(restored, folder / f'{key}.stl', tolerance=.03, angular_tolerance=.08)
                vertices, faces = restored.tessellate(.04, .08)
                meshes.append({'candidate': name, 'name': key, 'color': list(g.color),
                               'vertices_m': [[v.X/1000,v.Y/1000,v.Z/1000] for v in vertices],
                               'faces': faces})
                both.append(restored)
                if g.role != 'eye_shroud':
                    continue
                raw = primitive(g)
                uncut = decorative_shape(robot,g)
                added = volume(uncut.cut(raw))
                radius, depth = (v*1000 for v in g.size)
                expected = 0 if wall is None else math.pi*(radius-wall*1000)**2*(depth-wall*1000)
                removed = raw.volume - uncut.volume
                assert added < .0001 and abs(removed - expected) < .0001
                rows.append({'candidate':name,'part':key,'wall_mm':None if wall is None else wall*1000,
                             'volume_mm3':shape.volume,'removed_mm3':removed,'added_mm3':added,
                             'analytic_removed_mm3':expected,'step_roundtrip':'pass'})
            export_step(Compound(both),folder/'eyes.step')
            write(folder/'face-check.json', face_check)
            assert face_check['status']=='pass'
        selected = [r for r in rows if r['candidate']=='wall-2mm']
        removed = sum(r['removed_mm3'] for r in selected)
        extrusion = removed * assumptions['extruded_to_cad_volume_ratio']
        delta_material = extrusion*1e-9*assumptions['density_kg_m3']/assumptions['print_yield_fraction']*assumptions['filament_usd_per_kg']
        delta_machine = extrusion/assumptions['effective_extrusion_mm3_s']/3600/assumptions['print_yield_fraction']*assumptions['machine_usd_per_hour']
        report.update(status='partial',geometry_status='pass',rows=rows,
                      selected_wall_mm=2,removed_bilateral_mm3=removed,
                      assumed_bilateral_mass_reduction_g=removed*1e-6*assumptions['density_kg_m3'],
                      conditional_print_cost_reduction_usd=delta_material+delta_machine,
                      cost_basis='Unchanged illustrative process at 1000 or 5000 units; before contingency, no labor saving, not a quote or slice',
                      limitations=['Mounting and lighting remain unresolved; front cap remains opaque geometry',
                                   'No loaded strength, print fit, thermal or optical qualification',
                                   'No full composed assembly or derived mass model promotion',
                                   'Subtractive geometry cannot add occupied volume; existing head interference remains independent'])
        write(output/'meshes.json',meshes)
    except Exception as error:
        report['error']=str(error)
        (output/'error.log').write_text(traceback.format_exc())
    report['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts and p.name!='report.json'}
    write(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','output_hashes','rows')}))
    if report['status']=='fail': raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
