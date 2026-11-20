"""Compose selected source assemblies without moving or regenerating their parts.

Run with the repository .venv. Source CAD remains authoritative; this does not
create a connected electrical harness or a new physics model.
"""
from pathlib import Path
import copy, hashlib, json, sys, urllib.request
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/cove50-electronics-integration/r1-20261026'
DATA = ROOT / 'data/electronics-integration-r1-20261026'
POWER = ROOT / 'artifacts/cove50-power-integration/internal-r33-20261025/installed-torso-r25'
HEAD = ROOT / 'artifacts/cove50-compute-architecture/r23-powered-head-20261026'
SENSOR = ROOT / 'artifacts/cove50-sensor-integration/original-shell-layout-20261026'
RESULT = 'cove50-electronics-integrated-r1-20261026'
STUDY = 'cove50-electronics-integrated'

def read(p):
    return json.loads(p.read_text())

def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + '\n')

def relative(p):
    return str(p.relative_to(ROOT))

def witness(p):
    return dict(path=relative(p), bytes=p.stat().st_size,
                sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest())

def get_bom(result):
    with urllib.request.urlopen('http://127.0.0.1:8805/api/design-bom?result=' + result) as f:
        return json.load(f)

def world_mesh(scene, name):
    transform, key = scene.graph[name]
    mesh = scene.geometry[key].copy()
    mesh.apply_transform(transform)
    return mesh

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    power = read(POWER / 'parts.json')
    sensor = read(SENSOR / 'assembly-parts.json')
    head = read(HEAD / 'assembly-parts.json')
    pm = {p['name']: p for p in power['parts']}
    hm = {p['name']: p for p in head['parts']}
    old_shell = {'ape_body_shell__' + n for n in [
        'right_dark_eye', 'left_dark_eye', 'muzzle_shell', 'R_nostril_inlay',
        'L_nostril_inlay', 'mouth_inlay', 'face_mask', 'head_cranium_back',
        'head_cranium_front']}
    removed = {p['name'] for p in power['parts'] if p.get('owning_link') == 'head'} | old_shell
    overlap = (set(pm) - removed) & set(hm)
    common = []
    for name in sorted(overlap):
        a, b = pm[name], hm[name]
        # Shared neck mechanism must be the exact source and pose.
        if a.get('native_step') != b.get('native_step'):
            raise ValueError('Different shared neck source: ' + name)
        if not np.allclose(a['bounds_mm'], b['bounds_mm'], atol=1e-7):
            raise ValueError('Different shared neck pose: ' + name)
        common.append(dict(name=name, disposition='Retain once from power body',
                           native_step=a['native_step']))
    retained = [copy.deepcopy(p) for p in power['parts'] if p['name'] not in removed]
    added = [copy.deepcopy(p) for p in head['parts'] if p['name'] not in overlap]
    for p in retained:
        p['integration_source'] = 'power-r33-body'
    for p in added:
        p['source_owning_link'] = p.get('owning_link')
        p['owning_link'] = 'head'
        p['integration_source'] = 'accepted-compute-r23-head'
    parts = retained + added
    names = {p['name'] for p in parts}
    if len(names) != len(parts):
        raise ValueError('Duplicate composition name')
    body_sensor_comparison = []
    for p in sensor['parts']:
        name = p['name']
        if p.get('owning_link') == 'head' or name.startswith(('upper__', 'lower__')):
            continue
        if name == 'sensor_spine_integrated_v18':
            body_sensor_comparison.append(dict(name=name, disposition='Replaced once by R33 common power carrier'))
            continue
        if name not in pm:
            raise ValueError('Missing selected body sensor part: ' + name)
        changed = [k for k in ['native_step', 'mesh', 'bounds_mm', 'owning_link']
                   if p.get(k) != pm[name].get(k)]
        if changed:
            raise ValueError('Changed body sensor part: ' + name + str(changed))
        body_sensor_comparison.append(dict(name=name, disposition='Exact source and pose retained'))
    # No extra shell from V23: its three meshless inlay metadata records are
    # appearance reference, not fabricated head parts. R23 source skin is intact.
    ledger = dict(schema='cove50.electronics.ownership.v1',
                  source_counts=dict(power=len(pm), head=len(hm), sensor=len(sensor['parts'])),
                  removed_from_power=sorted(removed), shared_neck_retained_once=common,
                  body_sensor_comparison=body_sensor_comparison,
                  included_parts=len(parts), head_parts_added=len(added),
                  exclusions=['Old V22 head and generic reference head cladding',
                              'Old sensor spine and 29 displaced central-frame parts remain excluded by R33 ownership ledger',
                              'V23 removable_fan_bridge replaced by R23 combined_RF_and_regulator_bridge',
                              'Three meshless V23 mouth/nostril inlay reference entries are not extra manufactured parts',
                              'Unselected R24 head insulated-wire experiment'],
                  limitation='Static neutral-pose composition. No new all-body motion, fit, electrical continuity or physical qualification.')
    write(OUT / 'ownership-ledger.json', ledger)
    manifest = dict(schema='cove50.electronics.composed-parts.v1', parts=parts,
                    joints=sensor['joints'], controls=sensor['controls'],
                    root_link=sensor['root_link'], units=sensor['units'],
                    version='r1-20261026', scope='Selected complete-scene static CAD composition; original native part paths and poses retained.',
                    source_geometry_replacement=ledger['exclusions'])
    write(OUT / 'assembly-parts.json', manifest)
    scenes = [('assembly', 'assembly.glb'), ('inspection', 'inspection.glb')]
    for kind, power_file in ([] if '--metadata-only' in sys.argv else scenes):
        print('Loading ' + kind, flush=True)
        ps = trimesh.load(POWER / power_file, force='scene')
        hs = trimesh.load(HEAD / 'head-local-12V-routing.glb', force='scene')
        composed = trimesh.Scene()
        deviations = []
        for part in parts:
            name = part['name']
            source_scene = ps if part['integration_source'] == 'power-r33-body' else hs
            mesh = world_mesh(source_scene, name)
            if kind == 'inspection' and part['integration_source'] == 'accepted-compute-r23-head' and len(mesh.faces) > 20000:
                reduced = mesh.simplify_quadric_decimation(face_count=20000)
                deviation = float(np.max(abs(reduced.bounds - mesh.bounds)) * 1000)
                if deviation <= .1:
                    mesh = reduced
                    deviations.append(dict(name=name, bounds_deviation_mm=deviation))
            composed.add_geometry(mesh, node_name=name, geom_name=name)
        composed.export(OUT / (kind + '.glb'))
        print(kind, len(composed.graph.nodes_geometry), flush=True)
        if kind == 'inspection':
            write(OUT / 'display-detail.json', dict(scope='Display only, not native interference evidence', inherited_power_display=relative(POWER / power_file), new_head_mesh_reductions=deviations))
        del ps, hs, composed
    source_files = [POWER / 'parts.json', POWER / 'assembly.glb', POWER / 'inspection.glb',
                    SENSOR / 'assembly-parts.json', HEAD / 'assembly-parts.json',
                    HEAD / 'head-local-12V-routing.glb', HEAD / 'head-local-12V-routing.step',
                    ROOT / 'robot/electronics/sensing/selection.json',
                    ROOT / 'robot/electronics/head.json',
                    ROOT / 'robot/electronics/power/design.json',
                    ROOT / 'artifacts/cove50-compute-architecture/head-freeze-20261026/accepted-baseline.json',
                    ROOT / 'artifacts/cove50-power-integration/internal-r33-20261025/mounting-r2/frame-ownership-and-cell-revision-r1.json']
    write(OUT / 'source-lock.json', dict(schema='cove50.electronics.source-lock.v1', sources=[witness(p) for p in source_files], selected_versions=dict(sensor='v24', compute='accepted R23 / local-regulator-r2', power='r33', upper='v5', lower='v16'), source_authority='Selection pointers, not highest numbered experiment directory. No input geometry moved.'))
    # Snapshot current structured inventories and retain native purchasing units.
    inventories = {}
    for subsystem, rid in [('sensor', 'cove50-installed-sensing-v24'), ('compute', 'cove50-compute-head-local-regulator-r2-20261026'), ('power', 'cove50-internal-power-space-r33')]:
        inventories[subsystem] = get_bom(rid)
        write(OUT / ('source-' + subsystem + '-structured-bom.json'), inventories[subsystem])
    native_sensor = read(ROOT / 'artifacts/cove50-sensor-integration/selected-v24-20261026/sensor-bom.json')
    write(OUT / 'source-sensor-full-catalog.json', native_sensor)
    items = copy.deepcopy(inventories['power']['bom']['items'])
    excluded_bom = []
    native_by_id = {i['id']: i for i in native_sensor['items']}
    sensor_ids = set()
    for item in inventories['sensor']['bom']['items']:
        original = native_by_id.get(item['id'], {})
        associated = original.get('part_names', [])
        if associated and not set(associated) <= names:
            excluded_bom.append(dict(id=item['id'], part_names=associated, reason='Superseded or absent in composed CAD'))
            continue
        item = copy.deepcopy(item)
        item['id'] = 'sensor-' + item['id']
        items.append(item)
        sensor_ids.add(original.get('id'))
    # Full source catalog has compute/cooling and mounting detail absent from
    # the deliberately sensor-only dashboard BOM; transfer only present parts.
    for original in native_sensor['items']:
        if original['id'] in sensor_ids:
            continue
        associated = original.get('part_names', [])
        if not associated or not set(associated) <= names:
            excluded_bom.append(dict(id=original['id'], part_names=associated, reason='Superseded or absent in composed CAD'))
            continue
        q = original.get('installed_quantity', original.get('quantity', 1))
        price = original.get('buy_unit_price')
        unit_price = original.get('unit_cost')
        if unit_price is None and price is not None and original.get('category') == 'purchased':
            unit_price = price / original.get('units_per_buy_unit', 1)
        items.append(dict(id='catalog-' + original['id'], name=original['name'], quantity=q, unit='ea',
                          category=original.get('category', ''), part_number=original.get('part_number', ''),
                          material=original.get('material', ''), supplier=original.get('supplier', ''),
                          unit_cost=unit_price, currency=original.get('currency') or '',
                          cost_basis=original.get('cost_basis') or ('quote' if unit_price is not None else 'unknown'),
                          price_date=original.get('price_date') or original.get('checked_date') or '', unit_mass_g=original.get('unit_mass_g'),
                          source_refs=[dict(label='Original catalog, purchasing units and nominal CAD', path=relative(OUT / 'source-sensor-full-catalog.json'))],
                          notes=original.get('notes', '') + ' Buying units and price basis retained in source catalog; nominal attachment is not a delivered part.'))
    for original in inventories['compute']['bom']['items']:
        item = copy.deepcopy(original)
        item['id'] = 'compute-' + item['id']
        item['source_refs'] = item.get('source_refs') or [dict(
            label='Accepted compute increment source inventory',
            path=relative(OUT / 'source-compute-structured-bom.json'))]
        items.append(item)
    # The accepted head freeze is the authority for later dated module quotes;
    # the sensor session's inherited compute catalog can predate that freeze.
    frozen = read(ROOT / 'artifacts/cove50-compute-architecture/head-freeze-20261026/accepted-baseline.json')
    quote_owner = {'Central compute': 'catalog-compute-rock4d',
                   'Heatsink': 'catalog-compute-sink', 'Fan': 'catalog-compute-fan',
                   'GPIO receptacle': 'catalog-compute-gpio-mate'}
    by_id = {item['id']: item for item in items}
    for component in frozen['components']:
        target = quote_owner.get(component['role'])
        quote = component.get('quoted_price')
        if target and quote:
            item = by_id[target]
            item.update(unit_cost=quote['value'], currency=quote['currency'], cost_basis='quote')
            if quote.get('date'):
                item['price_date'] = quote['date']
            item['source_refs'].append(dict(label='Accepted head freeze component observation', path=relative(source_files[-2])))
    items.append(dict(id='compute-rf-pair', name='Molex 1461530150 antenna assemblies',
                      quantity=2, unit='ea', part_number='1461530150', category='purchased',
                      unit_cost=2.49, currency='EUR', cost_basis='estimate', unit_mass_g=None,
                      notes='Retained EUR4.98 PAIR observation divided by two for unit accounting. Two source antenna/coax assemblies installed, not four. Dated availability not reconfirmed.',
                      source_refs=[dict(label='Accepted compute component catalog', path=relative(source_files[-2]))]))
    for item in items:
        for field in ['currency', 'price_date', 'supplier', 'part_number', 'material', 'notes', 'category']:
            item[field] = item.get(field) or ''
    bom = dict(schema='robot-design-bom.v1', scope='One selected electronics composition: installed sensing, complete compute/cooling/RF head and power mounting, counted once.',
               completeness='partial', items=items,
               notes='Source estimates remain partial. Nine cell maxima and carrier density screens are not a complete robot mass. Stock/pack buying quantities remain in source catalogs; known BOM cost is not a purchase or landed quote. Storage remains candidate only, not installed. Old spine and fan bridge excluded; shared cameras and microphone appear once.',
               exclusions=['Full mechanical body actuator/shell purchasing inventory (present as CAD context)', 'Remaining power modules, connected cell/main/sense/NTC and moving harness', 'Storage card seating, all data cables/connectors and physical qualification', 'Tax, shipping, manufacturing labor and complete robot mass/inertia'])
    write(OUT / 'electronics-bom.json', bom)
    write(OUT / 'bom-ownership.json', dict(excluded_source_lines=excluded_bom,
          installed_storage=False, shared_sensor_modules='Owned once by transferred sensor inventory; compute baseline sensor list is provenance only.',
          power_carrier='Owned once by power BOM; sensor_spine_integrated_v18 excluded.',
          RF='Two antenna assemblies; EUR4.98 source observation is pair price.',
          completeness='Partial, includes known module and nominal attachment lines; source native catalog retains procurement units.'))
    interfaces = dict(schema='cove50.electronics.interfaces.v1',
        selected_head='Local Pololu2851 D24V50F5 input6..38V, nominal12V to5V; source output routing retained.',
        head_open_endpoints_mm=dict(VIN=[23,17,422], protected_return=[25.5,19.5,422]),
        body_supply='Protected body branch, return, connector and moving service loop unfinished. No fabricated body endpoint added.',
        electronics_output_reserve_W=dict(sustained=28, peak=45),
        reserve_accounting='One aggregate useful output budget; head converter, fan, compute and sensors consume it, not an extra second budget. No new duration estimate.',
        torso_converter='Old D42V110F5 host5V planning direction is not a second installed host regulator. Reconcile remaining body rail allocation in T-42fga9 with selected local head topology.',
        deferred=['Complete body/head protected supply and data/CSI/USB/I2S harness', 'Per-joint moving cable routing, strain relief, insulation and service', 'Thermal/loaded boot/RF/sensing and full motion qualification', 'Complete robot mass/inertia/actuator dynamics and compute latency in T-3753b7'],
        follow_up_tasks=['T-42fga9', 'T-3753b7'],
        qualification='CAD composition and inherited evidence only. An open electrical endpoint is not a connected design.')
    write(OUT / 'interfaces-and-open-work.json', interfaces)
    report = dict(schema='cove50.electronics.integration-report.v1',
        observation='Sensor V24, accepted R23 compute head and power R33 now share one neutral-pose complete robot scene. Obsolete head, spine and fan bridge are excluded; source axes and current cell/mount poses are retained.',
        measurements=dict(composed_parts=len(parts), source_power_parts=len(pm), removed_old_head_parts=len(removed), accepted_head_parts=len(hm), deduplicated_neck_parts=len(overlap), retained_body_sensor_records=len(body_sensor_comparison), source_joints=len(sensor['joints']), structured_bom_lines=len(items)),
        validation='Exact source-path/pose comparison for58 body sensor records and seven shared neck parts; unique853scene names. Cross-source native broad phase35pairs:23inheritedR23neckinterfaces,9validsolid zero common-volume observations,3invalid supplier-case bounds unresolved. Inherited within-headR23 and torsoR33 native/FEM/service evidence stays source-scoped. No new all-body motion, thermal or physical measurement. Native details in native-interface-observations.json.',
        limitations=interfaces['deferred'] + ['Source invalid motor-case geometry and nominal fit/tolerance uncertainty remain.', 'Unknown part masses prevent a complete total robot weight claim.'],
        source_versions=dict(sensor='v24', compute='accepted R23 / local-regulator-r2', power='r33'),
        follow_up='T-3753b7', integration_task='T-qne4ct')
    write(OUT / 'integration-report.json', report)
    write(DATA / 'composition.json', dict(result_id=RESULT, study_id=STUDY, report=witness(OUT / 'integration-report.json'), source_lock=witness(OUT / 'source-lock.json'), follow_up='T-3753b7'))
    print(json.dumps(report['measurements']), flush=True)

if __name__ == '__main__':
    main()
