"""Compare actual Pip face-lens Boolean STEP fidelity for ellipsoid pole orientations."""
import json
from pathlib import Path
from build123d import Pos, Rot, Sphere, export_step, import_step, scale
from lab.camera_mount import removal_clearance
from lab.config import load_config
from lab.mechanical import part_shape
from lab.robot import make_robot

OUT = Path('artifacts/ellipsoid-step-probe-v2')
OUT.mkdir(parents=True, exist_ok=True)
robot = make_robot(load_config(Path('variants/pip-head-service.yaml')))
head = next(x for x in robot.links if x.name == 'head')
g = next(x for x in head.geometry if x.role == 'face_lens')
placement = Pos(*(v * 1000 for v in g.pos)) * Rot(*(v * 180 / 3.141592653589793 for v in g.rpy))
def cutter():
    # Boolean operations may mutate OCCT topology owned by the operand.
    return removal_clearance(robot)
size = tuple(v * 1000 for v in g.size)
old = placement * scale(Sphere(1), by=size) - cutter()
new = placement * scale(Rot(0, 90, 0) * Sphere(1), by=size) - cutter()
# Export each as its own solid; source BRep and re-import volumes/bounds are the target evidence.
rows = {}
for name, shape in [('old_shared_primitive', old), ('pre_rotated_pole', new)]:
    path = OUT / (name + '.step')
    export_step(shape, path)
    back = import_step(path)
    rows[name] = {
        'source_volume_mm3': shape.volume,
        'roundtrip_volume_mm3': back.volume,
        'relative_volume_error': abs(back.volume - shape.volume) / shape.volume,
        'source_bounds_mm': [list(shape.bounding_box().min), list(shape.bounding_box().max)],
        'roundtrip_bounds_mm': [list(back.bounding_box().min), list(back.bounding_box().max)],
    }
rows['shared_part_shape_volume_mm3'] = part_shape(
    robot, head, next(i for i, x in enumerate(head.geometry) if x.role == 'face_lens')
).volume
(OUT / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows,indent=2))
