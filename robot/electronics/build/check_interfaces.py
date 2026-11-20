"""Native neutral-pose queries at the newly composed head/torso boundary."""
from pathlib import Path
import hashlib, json, time
import numpy as np
from build123d import import_step

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/cove50-electronics-integration/r1-20261026'

def main():
    manifest = json.loads((OUT / 'assembly-parts.json').read_text())
    parts = manifest['parts']
    source_head = json.loads((ROOT / 'artifacts/cove50-compute-architecture/r23-powered-head-20261026/assembly-parts.json').read_text())
    source_names = {p['name'] for p in source_head['parts']}
    new = [p for p in parts if p['integration_source'] == 'accepted-compute-r23-head']
    body = [p for p in parts if p['integration_source'] == 'power-r33-body']
    cache, trace = {}, []
    def load(p):
        path = ROOT / p['native_step']
        if str(path) not in cache:
            shape = import_step(path)
            cache[str(path)] = (shape, dict(path=p['native_step'],
                sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
                solids=len(shape.solids()), valid=shape.is_valid))
        return cache[str(path)]
    for a in new:
        if 'bounds_mm' not in a:
            continue
        for b in body:
            if 'bounds_mm' not in b:
                continue
            delta = np.minimum(a['bounds_mm'][1], b['bounds_mm'][1]) - np.maximum(a['bounds_mm'][0], b['bounds_mm'][0])
            if not (delta > 1e-6).all():
                continue
            row = dict(a=a['name'], b=b['name'], bounding_box_overlap_mm3=float(np.prod(delta)))
            if b['name'] in source_names:
                row['disposition'] = 'Same neck mechanism included in accepted R23 head; inherited source interface, not newly changed.'
            elif not a.get('native_step') or not b.get('native_step'):
                row['disposition'] = 'Native source unavailable; bounding boxes do not establish material contact.'
            else:
                start = time.monotonic()
                try:
                    sa, wa = load(a)
                    sb, wb = load(b)
                    row['sources'] = [wa, wb]
                    if not wa['valid'] or not wb['valid']:
                        row['disposition'] = 'Invalid supplier/source solid; retained bound uncertainty, no clearance inference.'
                    else:
                        overlap = sa.intersect(sb)
                        solids = [] if overlap is None else list(overlap.solids()) if hasattr(overlap, 'solids') else [s for x in overlap for s in x.solids()]
                        row['native_common_volume_mm3'] = sum(abs(s.volume) for s in solids)
                        row['common_solids'] = len(solids)
                        row['common_validity'] = [s.is_valid for s in solids]
                        row['disposition'] = 'Native material common-volume observation; neutral pose only.'
                except Exception as e:
                    row['disposition'] = 'Native query exception; unresolved.'
                    row['exception'] = repr(e)
                row['elapsed_s'] = time.monotonic() - start
            trace.append(row)
            print(row['a'], row['b'], row.get('native_common_volume_mm3', row['disposition']), flush=True)
    report = dict(schema='cove50.electronics.native-interface-observations.v1',
                  pose='Source neutral pose, untransformed native STEP in mm',
                  broad_phase_pairs=len(trace), observations=trace,
                  limitation='Only changed cross-source head/body boundary queries. No whole-body sweep, tolerances, electrical continuity, thermal or physical trial. Source invalidity remains an uncertainty, not omitted collision stock.')
    (OUT / 'native-interface-observations.json').write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()
