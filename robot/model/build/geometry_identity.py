"""Native correspondence of canonical FEM upper sheets to installed sheets."""
import numpy as np
from build123d import import_step, Plane, CenterOf
from scipy.spatial import cKDTree
from common import *

def main():
    parts = read(CAD / 'assembly-parts.json')['parts']
    rows = []
    keys = {'upper-F1': 'pitch_to_roll_one_piece', 'upper-F2': 'roll_to_yaw_one_piece',
            'upper-F3': 'yaw_to_elbow_one_piece', 'upper-F4': 'forearm_wrist_one_piece'}
    for block in read(OUT / 'compliance.json')['blocks']:
        if 'upper-F' not in block['id']:
            continue
        source = ROOT / block['source_parts'][0]['path']
        shape = import_step(source)
        side = block['id'][0]
        part = next(p for p in parts if p.get('source_name') == side + '_' + keys[block['id'][2:]])
        installed = import_step(ROOT / part['native_step'])
        if side == 'L':
            shape = shape.mirror(Plane.YZ)
        common = installed & shape
        volume = sum(s.volume for s in common.solids()) if common else 0.
        def face_rows(s):
            return np.array([list(f.center(CenterOf.MASS))+[f.area]+list(f.bounding_box().min)+list(f.bounding_box().max) for f in s.faces()])
        a, b = face_rows(shape), face_rows(installed)
        distance = max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())
        row = dict(block=block['id'], source_native=ref(source), selected_native=ref(ROOT / part['native_step']),
                   source_volume_mm3=shape.volume, selected_volume_mm3=installed.volume, common_volume_mm3=volume,
                   noncommon_source_mm3=shape.volume-volume, noncommon_selected_mm3=installed.volume-volume,
                   source_faces=len(a), selected_faces=len(b),
                   maximum_native_face_descriptor_distance=float(distance),
                   face_descriptor='Face mass centre(mm),area(mm²),AABB min/max(mm); symmetric nearest-neighbor comparison',
                   boolean_limit='Coincident mirrored F3/F4 compound common returned empty despite matching native face descriptors/volume/centroid/bounds; do not interpret that Boolean result as a physical non-overlap.',
                   COM_difference_mm=float(np.linalg.norm(np.array(tuple(shape.center(CenterOf.MASS))) -
                                                          np.array(tuple(installed.center(CenterOf.MASS))))))
        rows.append(row)
        print(row['block'], row['noncommon_source_mm3'], row['noncommon_selected_mm3'], flush=True)
    write(OUT / 'FEM-geometry-identity.json', dict(rows=rows,
        scope='Native manufactured-sheet comparison. Left polar/axial reflection applies to made upper sheets only, never supplier actuators.'))

if __name__ == '__main__':
    main()
