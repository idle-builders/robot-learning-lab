"""Source-bound conversion of retained design collections into variant/version records."""
import json,re,hashlib
from pathlib import Path
def enrich(c,h):
 for rec in c.execute("SELECT id,study_id,study_version FROM results WHERE record_kind='study'").fetchall():
  source=json.loads(c.execute('select payload_json from source_payload where result_id=?',(rec['id'],)).fetchone()[0])
  if 'legacy_collection' not in source:continue
  option=source['design'];cad=option.get('cad',{})
  study,version=rec['study_id'],rec['study_version']
  dated_identity=re.sub(r'-20\d{6}$','',option['id'])
  dated_revision=re.fullmatch(r'(.+?)[-_]r(\d+[a-z]?)',dated_identity,re.I)
  if dated_revision:
   study=source['legacy_collection']+':'+dated_revision[1];version='r'+dated_revision[2]
  if source['legacy_collection']=='cove50-hip-resume-20261020' and dated_identity=='t5l-anchor-a':
   study=source['legacy_collection']+':t5l-anchor-a';version='r1'
  if option['id'].endswith('-first-native'):
   base=option['id'].removesuffix('-first-native');study=source['legacy_collection']+':'+base
   m=re.search(r'/[ABC]-knee-r(\d+)/parts.json$',cad.get('parts_path',''));version='r'+m[1] if m else 'native-0'
  # A new mechanism architecture is its own variant, not the designer's next version.
  if source['legacy_collection']=='cove50-cad-competition-20261016':
   if option['id'] in ['CAD-hip-C-r0','CAD-hip-C-r1']:study=source['legacy_collection']+':CAD-hip-C-coupled-rod'
   elif option['id']=='CAD-hip-C-r2':study=source['legacy_collection']+':CAD-hip-C-bevel'
   elif option['id']=='CAD-hip-B-r0':study=source['legacy_collection']+':CAD-hip-B-belt'
   elif option['id']=='CAD-hip-B-r1':study=source['legacy_collection']+':CAD-hip-B-passive-core'
  if source['legacy_collection']=='cove50-ankle-working-foot-20261020' and option['id'] in ['ankle-A-local-serial','ankle-B-proximal-tendon','ankle-C-supported-serial']:
   manifests=[r['path'] for r in option.get('source_refs',[]) if r.get('path','').endswith('/design.json')]
   if manifests:version=Path(manifests[0]).parent.name
  c.execute('update results set study_id=?,study_version=? where id=?',(study,version,rec['id']))
  source['study_id']=study;source['version']=version
  evidence=[]
  for ref in option.get('source_refs',[]):
   path=ref.get('path','');label=ref.get('label',Path(path).name)
   if not path.endswith('.json') or any(t in Path(path).name for t in ['parts','freeze','hash','metadata']):continue
   metrics=[]
   try:
    native=json.loads((h.ROOT/path).read_text())
    if isinstance(native,dict):
     for key,value in native.items():
      if isinstance(value,(int,float)) and not isinstance(value,bool):
       unit=next((unit for suffix,unit in [('_kg','kg'),('_mm','mm'),('_nm','N m'),('_s','s'),('_deg','deg')]if key.lower().endswith(suffix)),None)
       if unit:metrics.append({'name':key,'value':value,'unit':unit,'condition':'Native report conditions'})
   except (OSError,ValueError):pass
   evidence.append({'id':'source-'+hashlib.sha256(path.encode()).hexdigest()[:10],'kind':'simulation' if re.search(r'forward|dynamic|gravity|load|mass.feedback',label+' '+path,re.I) else 'analysis','title':label,'observation':'Retained source evidence for this exact design revision.','measurements':metrics,'conditions':{'source_report':path},'limitations':'Interpret using the assumptions and limitations in the native report.','source_refs':[ref]})
  if evidence:option['evidence']=evidence
  section=c.execute('select position,payload_json from sections where result_id=? order by position',(rec['id'],)).fetchone();payload=json.loads(section['payload_json']);payload['groups'][0]['options']=[option]
  labels=h.design_sections(source.get('legacy_group',''))
  if labels!=['Whole robot']:
   payload['groups'][0]['robot_sections']=labels;c.execute('update results set robot_sections_json=? where id=?',(json.dumps(labels),rec['id']))
  c.execute('update sections set payload_json=? where result_id=? and position=?',(json.dumps(payload),rec['id'],section['position']))
  source['design']=option;c.execute('update source_payload set payload_json=? where result_id=?',(json.dumps(source),rec['id']))
 attach_owned_sources(c,h)
 attach_fixture_evidence(c,h)

def attach_fixture_evidence(c,h):
 """Attribute retained simulations by an embedded CAD path/hash, never by outcome."""
 parent='cove50-ankle-contact-fixture-20261020'
 rows=[dict(r) for r in c.execute('select * from assets where result_id=?',(parent,))]
 if not rows:return
 known={};families={}
 for row in c.execute("select r.id,r.study_id,s.payload_json from results r join source_payload s on s.result_id=r.id where r.record_kind='study'").fetchall():
  payload=json.loads(row['payload_json']);option=payload.get('design',{})
  manifests=[ref.get('path') for ref in option.get('source_refs',[]) if ref.get('path','').endswith('/design.json')]
  if (option.get('cad',{}).get('parts_path') or '').endswith('/design.json'):manifests.append(option['cad']['parts_path'])
  for path in manifests:
   known[path]=row['id'];families[str(Path(path).parent.parent)]=(row['study_id'],row['id'])
 manifest_hashes={}
 for path in (h.ROOT/'artifacts/cove50-engineering-competition/ankle-studio').rglob('design.json'):
  manifest_hashes[h.digest_file(path)]=str(path.relative_to(h.ROOT))
 reports={};direct={};directory_targets={}
 def identities(value):
  paths=set();hashes=set()
  def visit(v):
   if isinstance(v,dict):
    for k,item in v.items():
     if k.endswith('/design.json'):paths.add(k)
     if isinstance(item,str) and item.startswith('artifacts/') and item.endswith('/design.json'):paths.add(item)
     if 'design' in k and 'sha256' in k and isinstance(item,str):hashes.add(item)
     if isinstance(item,(dict,list)):visit(item)
   elif isinstance(v,list):
    for item in v:visit(item)
  visit(value)
  paths.update(manifest_hashes[digest] for digest in hashes if digest in manifest_hashes)
  return paths
 for asset in rows:
  if not asset['path'].endswith('.json') or asset['bytes']>8_000_000:continue
  try:data=json.loads((h.ROOT/asset['path']).read_text())
  except (OSError,ValueError):continue
  paths=identities(data)
  if len(paths)==1:
   path=next(iter(paths));direct[asset['path']]=path;reports[asset['path']]=data
   directory_targets.setdefault(str(Path(asset['path']).parent),set()).add(path)
 def source_for(path):
  if path in direct:return direct[path]
  directory=Path(path).parent
  while str(directory).startswith('artifacts/cove50-engineering-competition'):
   candidates=directory_targets.get(str(directory),set())
   if len(candidates)==1:return next(iter(candidates))
   directory=directory.parent
  return None
 def target_for(manifest):
  if manifest in known:return known[manifest]
  family=families.get(str(Path(manifest).parent.parent))
  if not family:return None
  study,latest=family;version=Path(manifest).parent.name
  existing=c.execute('select id from results where study_id=? and study_version=?',(study,version)).fetchone()
  if existing:known[manifest]=existing[0];return existing[0]
  source=h.ROOT/manifest
  if not source.is_file():return None
  native=json.loads(source.read_text());key=study+':'+version
  cad={};assets=[{'id':'manifest','title':'Exact native design','kind':'data','path':manifest}]
  match=re.search(r'/([ABC])/([^/]+)/design.json$',manifest)
  if match:
   neutral='artifacts/cove50-engineering-competition/ankle-studio/lead/'+match[1]+'-'+match[2]+'-neutral-native.glb'
   if (h.ROOT/neutral).is_file():cad={'model_units':'m','parts_path':manifest,'views':[{'label':'Source-matched neutral assembly','model_path':neutral}]};assets.append({'id':'neutral','title':'Source-matched neutral assembly','kind':'model','path':neutral})
  step=str(Path(manifest).with_name('assembly.step'))
  if (h.ROOT/step).is_file():assets.append({'id':'step','title':'Native assembly STEP','kind':'data','path':step});cad['step_path']=step
  title=str(native.get('design_id',f'Historical native foot {version}'))
  option={'id':key,'title':title,'summary':'Retained CAD revision with source-identified simulation evidence.','visual_kind':'native_CAD_source','maturity':'Historical engineering evidence; consult exact source limitations.','cad':cad,'source_refs':[{'label':'Exact design manifest','path':manifest}]}
  record={'id':key,'study_id':study,'version':version,'title':title,'dashboard':'design','workspace':'Body','robot_sections':['Ankle / foot'],'area':'Ankle / foot','observation':option['summary'],'measurements':'Native measurements are retained in attached simulation reports.','limitations':'Isolated fixtures do not establish whole-robot control or physical behavior.','assets':assets,'design':option,'provenance':[manifest],'clips':[]}
  h.save_result(c,record,[]);c.execute("update results set record_kind='study',collection_id=? where id=?",(parent,key));known[manifest]=key;return key
 attached={}
 for asset in rows:
  manifest=source_for(asset['path']);target=target_for(manifest) if manifest else None
  if not target:continue
  c.execute('insert or ignore into assets select ?,id,title,kind,path,sha256,bytes from assets where result_id=? and id=?',(target,parent,asset['id']))
  attached.setdefault(target,[]).append(asset)
 for clip in c.execute('select id,source_path from clips where result_id=?',(parent,)).fetchall():
  manifest=source_for(clip['source_path']);target=target_for(manifest) if manifest else None
  if target:c.execute('insert or ignore into design_study_clips values (?,?)',(target,clip['id']))
 for target,assets in attached.items():
  items=[]
  for asset in assets:
   if asset['path'] not in reports:continue
   data=reports[asset['path']];metrics=[]
   if isinstance(data,dict):
    for name,value in data.items():
     if isinstance(value,(int,float)) and not isinstance(value,bool):
      unit=next((unit for suffix,unit in [('_kg','kg'),('_mm','mm'),('_nm','N m'),('_s','s'),('_deg','deg')]if name.lower().endswith(suffix)),None)
      if unit:metrics.append({'name':name,'value':value,'unit':unit,'condition':'Exact native report conditions'})
   related=[{'label':a['title'],'path':a['path']} for a in assets if Path(a['path']).parent==Path(asset['path']).parent and any(t in Path(a['path']).name for t in ['trace','config','method'])]
   clips=[r[0] for r in c.execute('select sc.clip_id from design_study_clips sc join clips cl on cl.id=sc.clip_id where sc.result_id=? and cl.source_path like ?',(target,str(Path(asset['path']).parent)+'/%'))]
   items.append({'id':'fixture-'+hashlib.sha256(asset['path'].encode()).hexdigest()[:12],'title':asset['title'],'kind':'simulation','observation':'Retained native simulation associated through its exact CAD source path or source hash.','measurements':metrics,'conditions':{'source_design':direct[asset['path']]},'limitations':'Read the original report for fixture constraints, solver settings, early ends and physical transfer limits.','source_refs':[{'label':'Native report','path':asset['path']}]+related[:4],'clip_ids':clips})
  c.execute('insert or replace into sections values (?,?,?,?,?)',(target,900,'Source-bound simulations','Exact-source association; original multi-variant comparison retained in Archive.',json.dumps({'kind':'design_evidence','items':items})))
 return len(attached)

def attach_owned_sources(c,h):
 """Keep a variant's explicit source-card directory with its study version."""
 studies=[];owners={}
 for row in c.execute("select r.id,r.study_id,r.collection_id,s.payload_json from results r join source_payload s on s.result_id=r.id where r.record_kind='study'").fetchall():
  data=json.loads(row['payload_json']);option=data.get('design',{})
  directories={str(Path(ref['path']).parent) for ref in option.get('source_refs',[]) if ref.get('path') and Path(ref['path']).name in {'card.md','evidence-index.json','design.json'}}
  studies.append((dict(row),option,directories))
  for directory in directories:owners.setdefault(directory,set()).add(row['study_id'])
 for row,option,directories in studies:
  directories={d for d in directories if len(owners[d])==1}
  if not directories or not row['collection_id']:continue
  source_assets=[dict(a) for a in c.execute('select * from assets where result_id=?',(row['collection_id'],))]
  owned=[a for a in source_assets if any(a['path'].startswith(d+'/') for d in directories)]
  for asset in owned:c.execute('insert or ignore into assets select ?,id,title,kind,path,sha256,bytes from assets where result_id=? and id=?',(row['id'],row['collection_id'],asset['id']))
  if option.get('cad',{}).get('views'):continue
  models=[a for a in owned if a['kind']=='model' and str(Path(a['path']).parent) in directories]
  if not models:continue
  models.sort(key=lambda a:(not bool(re.search(r'neutral|closed|assembly',Path(a['path']).name,re.I)),a['path']))
  manifest=next((a['path'] for a in owned if Path(a['path']).name=='design.json' and str(Path(a['path']).parent) in directories),None)
  option['cad']={'views':[{'label':a['title'],'model_path':a['path']} for a in models], 'parts_path':manifest,'scope':'Retained source-owned CAD exports. Source manifests define any supported motion; otherwise inspect saved configurations.'}
  section=c.execute('select position,payload_json from sections where result_id=? and position=0',(row['id'],)).fetchone()
  payload=json.loads(section['payload_json']);payload['groups'][0]['options']=[option]
  c.execute('update sections set payload_json=? where result_id=? and position=0',(json.dumps(payload),row['id']))
  data=json.loads(row['payload_json']);data['design']=option;c.execute('update source_payload set payload_json=? where result_id=?',(json.dumps(data),row['id']))
