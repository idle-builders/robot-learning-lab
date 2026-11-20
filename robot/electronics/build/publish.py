"""Register one composed design study, retaining the three source inventories."""
from pathlib import Path
import hashlib, json, shutil, sqlite3

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/cove50-electronics-integration/r1-20261026'
DATA = ROOT / 'data/electronics-integration-r1-20261026'
RESULT = 'cove50-electronics-integrated-r1-20261026'
STUDY = 'cove50-electronics-integrated'

def read(p):
    return json.loads(p.read_text())

def main():
    assets = {}
    def register(path, label=None):
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
        assets[path] = dict(id='electronics-' + hashlib.sha256(path.encode()).hexdigest()[:20],
                            title=label or Path(path).name, path=path,
                            kind='model' if path.endswith('.glb') else 'image' if Path(path).suffix in ['.png', '.jpg', '.webp'] else 'data')
    db = sqlite3.connect(ROOT / 'data/results.sqlite')
    # Retain sealed source evidence and native parts as provenance, not extra
    # installed assemblies. Only the composed scene is offered as current CAD.
    source_ids = ['cove50-installed-sensing-v24', 'cove50-compute-head-local-regulator-r2-20261026',
                  'cove50-internal-power-space-r33', 'cove50-power-supervision-r3']
    source_inventory, historical_unmaterialized = [], []
    for rid in source_ids:
        for path, digest, size in db.execute('select path,sha256,bytes from assets where result_id=?', (rid,)):
            entry = dict(result_id=rid, path=path, sha256=digest, bytes=size)
            source_inventory.append(entry)
            if (ROOT / path).is_file():
                register(path)
            else:
                # Source sessions intentionally expired cheap intermediate
                # derivatives. Do not resurrect those as current assembly CAD.
                historical_unmaterialized.append(entry)
    (OUT / 'source-publication-inventories.json').write_text(json.dumps(dict(
        source_assets=source_inventory, unmaterialized_historical_assets=historical_unmaterialized,
        scope='Source-version inventories retained as provenance. Current composed native part/mesh dependencies are separately materialized and registered; obsolete intermediate derivatives are not reconstructed.'), indent=2) + '\n')
    for p in (ROOT / 'robot/electronics').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            q = OUT / 'authored-source' / p.relative_to(ROOT)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, q)
    for p in DATA.rglob('*'):
        if p.is_file() and p.suffix in ['.json', '.md', '.csv'] and p.name not in ['source-commit.json', 'preservation-receipt.json', 'verification.json']:
            q = OUT / 'coordination' / p.relative_to(DATA)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, q)
    for p in OUT.rglob('*'):
        if p.is_file() and p.name != 'publication.json':
            register(str(p.relative_to(ROOT)))
    report = read(OUT / 'integration-report.json')
    limits = 'Static composition of selected source CAD. Remaining protected body/head supply, connected power/data harness, storage seating, thermal, motion and physical qualification remain open. Invalid supplier motor-case geometry limits three boundary observations. No complete robot mass or new runtime claim.'
    refs = [dict(label=label, path=str((OUT / file).relative_to(ROOT))) for label, file in [
        ('Exact selected source identities', 'source-lock.json'),
        ('Part replacement and shared ownership', 'ownership-ledger.json'),
        ('Power interfaces and unfinished work', 'interfaces-and-open-work.json'),
        ('Complete scene at full source mesh detail', 'assembly.glb'),
        ('Native head/torso boundary observations', 'native-interface-observations.json'),
        ('BOM source ownership', 'bom-ownership.json'),
        ('Registered learning-model follow-up', 'coordination/model-follow-up.json')]]
    if (OUT / 'coordination/review/review.json').exists():
        refs.append(dict(label='Independent composition review', path=str((OUT / 'coordination/review/review.json').relative_to(ROOT))))
    grades = dict(dimensions=[
        dict(dimension='Source composition and ownership', grade=4.1, scale=5, confidence='high', reason='Pinned selected V24/R23/R33 sources; 853 named meshes, shared neck and modules once, obsolete structure excluded.'),
        dict(dimension='Static head/torso fit evidence', grade=3.2, scale=5, confidence='medium', reason='Nine valid native cross-boundary observations have zero common volume; three invalid supplier-case observations remain unresolved. No motion or tolerance envelope.'),
        dict(dimension='Complete electrical installation', grade=2.1, scale=5, confidence='high', reason='Selected modules and local output routing represented; protected body supply branch, remaining power functions and full harness unfinished.'),
        dict(dimension='Mass and dynamics readiness', grade=1.8, scale=5, confidence='low', reason='Partial source masses and catalog dynamics; total robot mass/inertia, compliance, delays and identified actuator response belong to registered follow-up.'),
        dict(dimension='Ape appearance continuity', grade=3.6, scale=5, confidence='medium', reason='Accepted original compact head with no rear extension and selected local4mm torso relief composed in opaque renders; fitted moving shell unqualified.')])
    record = dict(id=RESULT, study_id=STUDY, version='r1-20261026', title='Integrated electronics',
        dashboard='design', workspace='Body', robot_sections=['Electronics', 'Whole robot'],
        area='Selected sensing, compute and power composition', status='user_selected_design_round',
        observation=report['observation'], measurements='853 source parts;22 retained source axes; sensingV24 / accepted computeR23 / powerR33;311 partial BOM lines.',
        limitations=limits, grades=grades, provenance=[str((OUT / 'source-lock.json').relative_to(ROOT))], clips=[], sections=[],
        assets=list(assets.values()), design=dict(id=RESULT, title='Integrated electronics',
            summary='The latest selected sensor, computer and power sessions in one robot CAD assembly.',
            description=report['observation'], selected=True, selected_on='2026-10-26',
            visual_kind='native_CAD_assembly', maturity='Selected source composition; remaining installation and physical qualification open',
            image_path=str((OUT / 'renders/electronics-cutaway.png').relative_to(ROOT)),
            cad=dict(model_units='m', motion_mode='saved_configurations',
                views=[dict(label='Complete selected robot · static neutral pose', model_path=str((OUT / 'inspection.glb').relative_to(ROOT)))],
                parts_path=str((OUT / 'assembly-parts.json').relative_to(ROOT)), step_path=None,
                report_path=str((OUT / 'integration-report.json').relative_to(ROOT)),
                has_body_reference=True, body_panel_label='Ape shell · selected local4mm torso relief',
                scope='One853part neutral scene. Native per-part STEP/BREP paths retained in manifest and assets; accepted full head STEP retained as source. Full source-detail GLB in sources. Display mesh reduction only. Source22joint metadata is retained for the next learning-model task; continuous movement is disabled for this static composition.'),
            source_refs=refs, bom=read(OUT / 'electronics-bom.json'), evidence=[dict(
                id='composition-r1', kind='analysis', title='Selected source composition',
                observation=report['observation'],
                measurements=[dict(name='Composed scene', value=853, unit='parts'), dict(name='Shared neck records counted once', value=7, unit='parts'), dict(name='Superseded head records removed', value=81, unit='parts'), dict(name='Native valid boundary queries', value=9, unit='pairs'), dict(name='Invalid supplier-case boundary uncertainty', value=3, unit='pairs')],
                conditions=dict(pose='Exact source neutral pose', sources='SensorV24 geometryV23; computeacceptedR23; powerR33; upperV5/lowerV16'),
                limitations=limits, source_refs=refs[:3] + refs[4:6], image_groups=[dict(title='Composed appearance and electronics inspection', images=[
                    dict(path=str((OUT / 'renders/complete-ape.png').relative_to(ROOT)), caption='Opaque source ape shell; accepted compact head, local4mm torso relief. Static pose.'),
                    dict(path=str((OUT / 'renders/electronics-cutaway.png').relative_to(ROOT)), caption='Head and torso skin hidden for inspection; actual selected modules and mounts. Missing supply/data harness and power functions remain open.')])])]))
    (OUT / 'publication.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(result=RESULT, assets=len(assets), publication=str((OUT / 'publication.json').relative_to(ROOT)))))

if __name__ == '__main__':
    main()
