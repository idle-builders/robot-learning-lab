"""Probe STEP fidelity for Boolean-cut nonuniform ellipsoids."""
from pathlib import Path
from build123d import Box, Pos, Rot, Sphere, export_step, import_step, scale

OUT = Path('artifacts/ellipsoid-step-probe-v1')
OUT.mkdir(parents=True, exist_ok=True)
SIZE = (7.0, 57.33, 24.6)

def check(name, shape):
    path = OUT / f'{name}.step'
    export_step(shape, path)
    back = import_step(path)
    return {
        'source_volume_mm3': shape.volume,
        'roundtrip_volume_mm3': back.volume,
        'source_bounds_mm': [list(shape.bounding_box().min), list(shape.bounding_box().max)],
        'roundtrip_bounds_mm': [list(back.bounding_box().min), list(back.bounding_box().max)],
        'relative_volume_error': abs(back.volume - shape.volume) / shape.volume,
        'max_bound_error_mm': max(abs(a-b) for a,b in zip(shape.bounding_box().min, back.bounding_box().min)) + max(abs(a-b) for a,b in zip(shape.bounding_box().max, back.bounding_box().max)),
    }

sphere = Sphere(1)
cut_world = Pos(5, 0, 0) * Box(5, 100, 100)
uncut = scale(sphere, by=SIZE)
cut_after_scale = uncut - cut_world
# Transform the cutter into unit-sphere coordinates, cut there, then apply scale.
cut_unit = Pos(5 / SIZE[0], 0, 0) * Box(5 / SIZE[0], 100 / SIZE[1], 100 / SIZE[2])
cut_before_scale = scale(sphere - cut_unit, by=SIZE)
results = {name: check(name, shape) for name, shape in [('uncut', uncut), ('cut_after_scale', cut_after_scale), ('cut_before_scale', cut_before_scale)]}
# Rotating the unit sphere before applying the affine scale moves its pole/seam
# away from the cut's problematic direction and preserves the exact cut on STEP
# round trip in the current OCCT/build123d stack.
rotated_pole = scale(Rot(0, 90, 0) * Sphere(1), by=SIZE) - cut_world
results['rotated_pole_before_scale'] = check('rotated_pole_before_scale', rotated_pole)
(OUT / 'results.json').write_text(__import__('json').dumps(results, indent=2) + '\n')
print(__import__('json').dumps(results, indent=2))
