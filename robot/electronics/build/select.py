"""Select the composed study and replace overlapping electronics BOM entries."""
from pathlib import Path
import json, urllib.request

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/electronics-integration-r1-20261026'
RESULT = 'cove50-electronics-integrated-r1-20261026'
STUDY = 'cove50-electronics-integrated'

def get(route):
    with urllib.request.urlopen('http://127.0.0.1:8805/api/' + route) as f:
        return json.load(f)

def post(route, data):
    request = urllib.request.Request('http://127.0.0.1:8805/api/' + route,
        data=json.dumps(data).encode(), headers={'Content-Type': 'application/json', 'Origin': 'http://127.0.0.1:8805'})
    with urllib.request.urlopen(request) as f:
        return json.load(f)

def main():
    before = get('design-selection')
    post('design-selection', dict(section='Electronics', study_id=STUDY))
    original = get('robot-bom')
    plan = original['plan']
    sources = {'cove50-installed-sensing-v24', 'cove50-compute-head-local-regulator-r2-20261026', 'cove50-internal-power-space-r33'}
    removed = [a for a in plan['assemblies'] if a['result_id'] in sources]
    plan['assemblies'] = [a for a in plan['assemblies'] if a['result_id'] not in sources | {RESULT}]
    plan['assemblies'].append(dict(result_id=RESULT, count=1))
    old = 'Selected complete-body sensing added once; six FSRs and all sensor modules already counted in this assembly. Compute context is not selected here.'
    note = 'Selected integrated electronics added once: sensorV24, accepted computeR23/local-regulator-r2 and powerR33. Separate sensor/compute/power assembly entries are replaced to avoid duplicates. Mechanical source body is context, not an extra mechanical purchase. Partial mass/cost; remaining power/data harness and physical qualification open.'
    plan['notes'] = plan.get('notes', '').replace(old, '').strip()
    if note not in plan['notes']:
        plan['notes'] += '\n' + note
    saved = post('robot-bom', dict(expected_revision=original['revision'], plan=plan))
    record = dict(schema='cove50.electronics.selection-consumption.v1',
                  preference_before=before, preference_after=get('design-selection'),
                  robot_BOM_revision_before=original['revision'], robot_BOM_revision_after=saved['revision'],
                  removed_overlapping_assemblies=removed, plan_after=saved['plan'])
    (DATA / 'selection-consumption.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(preferred=STUDY, robot_BOM_revision=saved['revision'], replaced=removed)))

if __name__ == '__main__':
    main()
