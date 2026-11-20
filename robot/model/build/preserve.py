"""Select exact model inputs/evidence for verified private preservation; never evict."""
import argparse,gzip,shutil,sqlite3
from common import *

def build():
    db=sqlite3.connect(ROOT/'data/blob-store.sqlite');db.row_factory=sqlite3.Row
    indexed=[dict(r)for r in db.execute('select files.*,packs.verified_at from files join packs on files.pack_key=packs.key')]
    by_path={r['path']:r for r in indexed};by_hash={r['sha256']:r for r in indexed if r['verified_at']}
    paths={p for p in OUT.rglob('*')if p.is_file() and 'recovery'not in p.relative_to(OUT).parts}
    # Capture authored sources and compact decisions without recursive snapshots.
    for folder in [ROOT/'robot/model',DATA]:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__'not in p.parts and not p.name.startswith('preservation-'):
                q=OUT/'recovery/frozen-control'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);paths.add(q)
    def refs(value):
        if isinstance(value,dict):
            for v in value.values():yield from refs(v)
        elif isinstance(value,list):
            for v in value:yield from refs(v)
        elif isinstance(value,str) and value.startswith(('artifacts/','robot/','artifacts/policies/checkpoints/')):
            p=ROOT/value
            if p.is_file():yield p
    for name in ['source-lock.json','part-mass-ledger.json','compliance.json','FEM-geometry-identity.json','actuators.json','sensors.json','trajectory-mapping.json']:
        paths.update(refs(read(OUT/name)))
    # Source FEM report, unit-field and fixture metadata identities stay linked.
    direct=list(paths)
    for p in direct:
        if p.suffix=='.json' and not p.is_relative_to(OUT):
            try:paths.update(refs(read(p)))
            except (ValueError,UnicodeDecodeError):pass
    paths.update(OUT/'recovery'/p for p in ['model-source.bundle'] if (OUT/'recovery'/p).is_file())
    selected={};mapping=[];unarchived_hashes={}
    for logical in sorted(paths):
        physical=logical.resolve();digest=sha(physical);size=physical.stat().st_size
        if not physical.is_relative_to(ROOT):raise ValueError(str(logical))
        # Reuse verified same bytes even when reached under another source name.
        previous=by_hash.get(digest)
        if previous and previous['bytes']==size:
            target=previous['path'];row=dict(path=target,bytes=size,mtime_ns=previous['mtime_ns'],sha256=digest)
        elif digest in unarchived_hashes:
            target=unarchived_hashes[digest];row=selected[target]
        else:
            target=rel(physical)
            if not target.startswith(('artifacts/','studies/','assets/','.artifact-store/objects/sha256/')):
                q=OUT/'recovery/source-inputs'/target;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(physical,q);physical=q;target=rel(q)
            if target in by_path and by_path[target]['sha256']!=digest:
                physical=OUT/'recovery/current-content'/digest/physical.name;physical.parent.mkdir(parents=True,exist_ok=True)
                if not physical.exists():shutil.copyfile(logical,physical)
                target=rel(physical)
            st=physical.stat();row=dict(path=target,bytes=size,mtime_ns=st.st_mtime_ns,sha256=digest);unarchived_hashes[digest]=target
        selected[target]=row;mapping.append(dict(logical_path=rel(logical),archive_path=target,sha256=digest,bytes=size))
    # Already verified files need not be re-uploaded; archive only materialized new entries.
    mapping_path=OUT/'recovery/logical-to-archive-paths.json'
    write(mapping_path,dict(schema='cove50.model.archive-map.v1',files=mapping,restoration='Restore archive_path then verify/copy bytes to logical_path; current native model and meshes remain local.'))
    mapping_digest=sha(mapping_path)
    if rel(mapping_path)in by_path and by_path[rel(mapping_path)]['sha256']!=mapping_digest:
        alias=OUT/'recovery/current-content'/mapping_digest/mapping_path.name
        alias.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(mapping_path,alias);mapping_path=alias
    st=mapping_path.stat();selected[rel(mapping_path)]=dict(path=rel(mapping_path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=mapping_digest)
    new=[r for r in selected.values()if not (r['path']in by_path and by_path[r['path']]['sha256']==r['sha256'] and by_path[r['path']]['verified_at'])]
    plan=dict(schema='robot-learning-lab.cold-selection.v1',reason='Current selected-CAD learning/perception model, full native episodes, exact preceding diagnostics and supplier/FEM inputs. No local eviction.',files=sorted(new,key=lambda r:r['path']))
    with gzip.open(DATA/'preservation-selection.json.gz','wt')as f:json.dump(plan,f)
    write(DATA/'preservation-content-inventory.json',dict(files=sorted(selected.values(),key=lambda r:r['path']),logical_files=len(mapping),new_files=len(new),logical_to_archive_map=ref(mapping_path),local_eviction=False))
    print(dict(logical_files=len(mapping),distinct_payloads=len(selected),new_files=len(new),new_bytes=sum(r['bytes']for r in new)))

def audit():
    inventory=read(DATA/'preservation-content-inventory.json');db=sqlite3.connect(ROOT/'data/blob-store.sqlite');db.row_factory=sqlite3.Row
    issues=[];packs=set()
    for f in inventory['files']:
        r=db.execute('select files.*,packs.verified_at from files join packs on files.pack_key=packs.key where path=?',(f['path'],)).fetchone()
        if not r or r['sha256']!=f['sha256']or r['bytes']!=f['bytes']or not r['verified_at']:issues.append(f['path'])
        else:packs.add(r['pack_key'])
    report=dict(schema='cove50.model.private-preservation.v1',files=len(inventory['files']),logical_files=inventory['logical_files'],bytes=sum(r['bytes']for r in inventory['files']),verified_packs=len(packs),content_identity_issues=issues,local_eviction=False,
        verification='SHA/byte identity against verified private Infomaniak pack index. New packs receive complete remote readback and sampled actual range/decompression restoration. Existing identical verified bytes reused. Exact logical-to-archive map retained.')
    write(DATA/'preservation-receipt.json',report);print(report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['build','audit']);a=parser.parse_args();build()if a.operation=='build'else audit()
