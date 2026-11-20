"""Freeze input identities and derive per-part CAD volume inertias.

Native homogeneous moments describe shape, not hidden motor/PCB internals.
Unknown purchasing masses use explicitly bounded material estimates.
"""
import collections
import sys
import numpy as np
import trimesh
from common import *

def shell_owner(name, centre):
    s = 'R' if centre[0] > 0 else 'L'
    if any(x in name for x in ['thigh', 'hip_flex']): return s + '_roll_link'
    if any(x in name for x in ['shin', 'knee_cuff', 'knee_flex']): return s + '_shin'
    if any(x in name for x in ['sole', 'smooth_boot', 'ankle_cuff', 'ankle_flex']): return s + '_foot'
    if any(x in name for x in ['upper_arm', 'shoulder_trim']): return s + '_upper_arm'
    if any(x in name for x in ['forearm', 'wrist', 'smooth_paw', 'elbow_cuff']): return s + '_forearm'
    return 'torso'

def estimate_mass(part, volume):
    n, role, material = part['name'], part.get('role', ''), part.get('material', '') or ''
    text = (n + ' ' + role + ' ' + material).lower()
    if part.get('mass_kg') is not None:
        m = float(part['mass_kg'])
        return m, [m * .9, m * 1.1], 'Retained source mass allocation; source basis in original record', 'source_allocation'
    if role == 'source_maximum_can_envelope':
        return .071, [.068, .071], 'Supplier 71 g maximum per P50B; lower 68 g is an unmeasured sensitivity assumption', 'supplier_maximum'
    if role == 'installation_size_scenario_not_additive_material':
        return 0., [0., 0.], 'Installation keep-out is not additive material', 'excluded_reservation'
    # Explicit module masses inherited from accepted component catalog. Never
    # turn a board enclosure box into homogeneous metal.
    known = {'ROCK4D_25x25_heatsink': .00907, 'ROCK4D_40mm_fan': .0138}
    if n in known:
        m = known[n]; return m, [m*.9, m*1.1], 'Accepted head component catalog', 'source_allocation'
    if 'alsi10mg' in text: density, spread, basis = 2670., .15, 'LPBF AlSi10Mg nominal density; process unmeasured'
    elif any(x in text for x in ['aluminium', 'aluminum', '5052', '6061']): density, spread, basis = 2700., .10, 'Nominal aluminium density'
    elif 'peek' in text: density, spread, basis = 1300., .15, 'Nominal PEEK density'
    elif any(x in text for x in ['pa12', 'nylon12', 'nylon', 'polyamide']): density, spread, basis = 1010., .25, 'Nominal PA12/nylon density; process/porosity unmeasured'
    elif 'tpu' in text or 'silicone' in text: density, spread, basis = 1150., .25, 'Elastomer density assumption; exact compound unmeasured'
    elif '5413' in text: density, spread, basis = 1420., .25, 'Polyimide/adhesive effective density assumption'
    elif 'nomex' in text: density, spread, basis = 980., .25, 'Nomex effective density assumption'
    elif '467mp' in text or 'adhesive' in text: density, spread, basis = 1000., .30, 'Adhesive effective density assumption'
    elif 'solder' in text or 'sac305' in text: density, spread, basis = 7370., .30, 'Solder process allowance, not measured deposited material'
    elif 'steel' in text or role in ['fastener', 'fastener_proxy', 'standard_fastener_reference', 'nominal_fastener', 'standard_nominal', 'nominal_hardware', 'source_dimension_fastener_bound', 'supplier_fastener_containment']:
        density, spread, basis = 7850., .20, 'Nominal steel hardware density; threads/coatings omitted'
    elif role in ['original_aesthetic_reference', 'ape_shell_reference', 'body_shell']:
        density, spread, basis = 1010., .50, 'Reference skin PA12 scenario; material, thickness and attachment not qualified'
    elif role in ['purchased_module', 'supplier_native', 'supplier_native_subset', 'source_power_board', 'native_routed_PCB_with_source_fuse_bound', 'native_routed_PCB_and_source_package_containment'] or material in ['Source part', 'Authentic supplier assembly', 'Manufacturer assembled module']:
        density, spread, basis = 2200., .70, 'Mixed populated-electronics effective density assumption; no delivered module mass'
    elif 'wire' in text or 'harness' in text or 'coax' in text:
        density, spread, basis = 3000., .60, 'Mixed conductor/insulation effective density assumption'
    else: density, spread, basis = 1200., .70, 'Unspecified material placeholder density; requires weighing or material identification'
    m = max(0., volume) * density
    return m, [m*(1-spread), m*(1+spread)], basis, 'material_estimate'

def main():
    m = read(CAD / 'assembly-parts.json')
    selected = read(ROOT / 'robot/electronics/selection.json')
    for record in selected['source'].values():
        if sha(ROOT / record['path']) != record['sha256']: raise ValueError('Selected input hash drift')
    print('Loading frozen complete scene', flush=True)
    scene = trimesh.load(CAD / 'assembly.glb', force='scene')
    catalog = read(CAD/'source-sensor-full-catalog.json')['items']
    catalog_material = {n: i['material'] for i in catalog if i.get('material') for n in i.get('part_names', [])}
    cache = OUT / 'native-properties-cache.json'
    props = read(cache) if cache.exists() else {}
    rows = []
    from build123d import import_step, CenterOf
    for index, part in enumerate(m['parts']):
        part = dict(part)
        if part.get('material') in [None, '', 'Source part'] and part['name'] in catalog_material:
            part['material'] = catalog_material[part['name']]
        name = part['name']; T, key = scene.graph[name]
        mesh = scene.geometry[key].copy(); mesh.apply_transform(T)
        centre = mesh.bounds.mean(0)
        native = ROOT / part['native_step'] if part.get('native_step') else None
        digest = sha(native) if native else None
        method = None
        if digest in props:
            q = props[digest]
        elif native:
            try:
                shape = import_step(native)
                # Supplier invalid parts are kept visible but cannot supply a
                # reliable homogeneous inertia by a signed volume integral.
                if not shape.is_valid or shape.volume <= 0: raise ValueError('Invalid/nonpositive native solid')
                vol = float(shape.volume)
                q = dict(volume_m3=vol*1e-9, com_CAD_m=(np.array(tuple(shape.center(CenterOf.MASS)))/1000).tolist(),
                         inertia_per_kg_CAD_m2=(np.array(shape.matrix_of_inertia)/vol/1e6).tolist(),
                         method='Native valid solid homogeneous volume moments')
            except Exception as e:
                ext = mesh.extents
                q = dict(volume_m3=abs(float(mesh.volume)) if mesh.is_watertight else float(mesh.convex_hull.volume),
                         com_CAD_m=centre.tolist(), inertia_per_kg_CAD_m2=np.diag((ext@ext-ext**2)/12).tolist(),
                         method='AABB inertia and centre; mesh/convex volume fallback', native_error=str(e))
            props[digest] = q
        else:
            # Original reference shell meshes have no fabricated STEP identity.
            if mesh.is_volume and mesh.volume > 0:
                q = dict(volume_m3=float(mesh.volume), com_CAD_m=mesh.center_mass.tolist(),
                         inertia_per_kg_CAD_m2=(mesh.moment_inertia/mesh.volume).tolist(), method='Reference watertight mesh homogeneous moments')
            else:
                # Thin reference surfaces: surface area times 1.2 mm is an
                # explicit shell estimate, never a filled convex-body mass.
                ext = mesh.extents
                q = dict(volume_m3=float(mesh.area*.0012), com_CAD_m=centre.tolist(),
                         inertia_per_kg_CAD_m2=np.diag((ext@ext-ext**2)/12).tolist(), method='Reference open skin: 1.2 mm area estimate with AABB inertia')
        q = dict(q)
        # A native compound may contain a few invalid supplier solids while
        # retaining a reported positive component volume. Do not fill all its
        # external gaps with material by using convex volume for its mass.
        source_vol = part.get('native_volume_mm3', part.get('volume_mm3'))
        if q['method'].startswith('AABB') and source_vol and source_vol > 0:
            q['fallback_enclosure_volume_m3'] = q['volume_m3']
            q['volume_m3'] = source_vol * 1e-9
            q['volume_basis'] = 'Retained native volume, including invalid-source uncertainty; AABB is used only for moments'
        mass, interval, basis, category = estimate_mass(part, q['volume_m3'])
        owner = part.get('owning_link') or 'torso'
        owner_basis = 'Original link ownership' if part.get('owning_link') else 'Torso-fixed R33 power mounting; authoring omitted owning_link'
        if part.get('role') == 'original_aesthetic_reference':
            owner = shell_owner(name, centre)
            owner_basis = 'Articulated visual/reference mass binding inferred from anatomical name and centroid sign; source static scene marked torso'
        row = dict(name=name, owning_link=owner, source_owning_link=part.get('owning_link'), ownership_basis=owner_basis,
                   mass_kg=mass, sensitivity_interval_kg=interval, mass_basis=basis, mass_category=category,
                   **q, native_source=ref(native) if native else None,
                   source_role=part.get('role'), source_material=part.get('material'),
                   source_reference_flag=part.get('inspection_reference'),
                   source_mass_kg=part.get('mass_kg'), bounds_CAD_m=mesh.bounds.tolist(),
                   source_mesh_sha256=hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest())
        rows.append(row)
        if index % 50 == 0:
            write(cache, props); print(index, name, flush=True)
    write(cache, props)
    # Module source quotes can include fan/heatsink masses under different
    # names. Reconcile them once via the selected full catalog's part mappings.
    by_name = {r['name']: r for r in rows}
    for item in catalog:
        names = item.get('part_names', [])
        unit_mass = item.get('unit_mass_g')
        if unit_mass is not None and len(names)==1 and names[0] in by_name:
            row = by_name[names[0]]
            if row['source_mass_kg'] is None:
                mass = float(unit_mass)/1000*float(item.get('installed_quantity',item.get('quantity',1)))
                row.update(mass_kg=mass, sensitivity_interval_kg=[mass*.9,mass*1.1], mass_basis='Accepted source catalog '+item['id'], mass_category='source_allocation')
        elif unit_mass is not None and names and all(n in by_name for n in names):
            total = sum(by_name[n]['mass_kg'] for n in names)
            target = float(unit_mass)/1000*float(item.get('installed_quantity',item.get('quantity',1)))
            for name in names:
                row = by_name[name]
                mass = target * row['mass_kg']/total
                row.update(mass_kg=mass, sensitivity_interval_kg=[mass*.85,mass*1.15],
                           mass_basis='Single assembly mass allocated by nominal material weights: '+item['id'], mass_category='source_allocation')
    write(OUT/'part-mass-ledger.json', dict(schema='cove50.part-mass-ledger.v1', parts=rows,
        interpretation='One record per composed part. Source reference flags from incremental BOMs do not mean the selected compute/sensors are absent. Reference skin estimates are explicit simulation scenarios.',
        incomplete_geometry=['Remaining power functions and cell/main/sense/NTC/data/moving harness', 'Unselected hand/wrist mechanisms and their sensors', 'Unrepresented consumables, hardware and fabrication changes'],
        interval_meaning='Independent engineering sensitivity bounds, not confidence intervals or manufacturing tolerances; no whole-robot measured weight.'))
    write(OUT/'frames.json', frame_contract())
    lock_sources = [ref(CAD/f) for f in ['assembly-parts.json','assembly.glb','source-lock.json','electronics-bom.json','source-sensor-full-catalog.json']]
    lock_sources += [ref(ROOT/f) for f in ['robot/electronics/selection.json','robot/electronics/sensing/selection.json']]
    write(OUT/'source-lock.json',dict(schema='cove50.learning-model.inputs.v1', sources=lock_sources,
        selected_versions=read(CAD/'source-lock.json')['selected_versions'], source_commit='ebc1c31181786604dea13578112c38c8850e0858',
        qualification='Model of exact selected design-round geometry, with explicit unmeasured physical assumptions; no new whole-robot mechanical selection.'))
    print(dict(parts=len(rows), mass_kg=sum(r['mass_kg'] for r in rows), mass_categories=dict(collections.Counter(r['mass_category'] for r in rows))), flush=True)

if __name__=='__main__': main()
