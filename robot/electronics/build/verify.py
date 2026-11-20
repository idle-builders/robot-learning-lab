"""Observe the published selected version, source assets and consuming BOM."""
from pathlib import Path
import hashlib, json, sqlite3, urllib.request

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/electronics-integration-r1-20261026'
OUT = ROOT / 'artifacts/cove50-electronics-integration/r1-20261026'
RESULT = 'cove50-electronics-integrated-r1-20261026'

def get(route):
    with urllib.request.urlopen('http://127.0.0.1:8805/api/' + route, timeout=60) as f:
        return json.load(f)

def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

def main():
    live = get('result?id=' + RESULT)
    design = live['sections'][0]['payload']['groups'][0]['options'][0]
    bom = get('design-bom?result=' + RESULT)
    robot_bom = get('robot-bom')
    prefs = get('design-selection')
    db = sqlite3.connect(ROOT / 'data/results.sqlite')
    issues, rows = [], 0
    for path, sha, size in db.execute('select path,sha256,bytes from assets where result_id=?', (RESULT,)):
        p = ROOT / path
        if not p.is_file() or p.stat().st_size != size or digest(p) != sha:
            issues.append(path)
        rows += 1
    lock = json.loads((OUT / 'source-lock.json').read_text())
    source_drift = [s['path'] for s in lock['sources'] if digest(ROOT / s['path']) != s['sha256']]
    manifest = json.loads((OUT / 'assembly-parts.json').read_text())
    missing_parts = []
    for part in manifest['parts']:
        for field in ['native_step', 'native_brep', 'mesh']:
            if part.get(field) and not (ROOT / part[field]).is_file():
                missing_parts.append(dict(name=part['name'], field=field, path=part[field]))
    report = dict(schema='cove50.electronics.published-consumption.v1',
        result_id=RESULT, published_status=live['status'], published_assets=rows,
        registered_asset_identity_issues=issues, source_lock_drift=source_drift,
        missing_current_part_dependencies=missing_parts,
        part_records=len(manifest['parts']), source_axes=len(manifest['joints']),
        static_viewer=design['cad']['motion_mode'] == 'saved_configurations' and len(design['cad']['views']) == 1,
        selected_preference=prefs.get('Electronics'),
        pointer=json.loads((ROOT / 'robot/electronics/selection.json').read_text()),
        structured_BOM_lines=len(bom['lines']), partial_BOM_totals=bom['totals'],
        robot_BOM_revision=robot_bom['revision'], robot_BOM_composition=robot_bom['plan']['assemblies'],
        limitations='Identity and consuming-path verification. No binary engineering score, complete robot weight, physical dynamics or qualification claim.')
    (DATA / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['pointer', 'robot_BOM_composition']}), flush=True)

if __name__ == '__main__':
    main()
