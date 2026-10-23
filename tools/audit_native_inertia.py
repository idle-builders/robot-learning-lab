"""Compare retained Newton body inertia against the sealed MuJoCo tensor."""
import argparse,hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.artifact_store import snapshot_file
p=argparse.ArgumentParser();p.add_argument('loaded',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
raw=json.loads(a.loaded.read_text());iface=json.loads((a.bundle/'model/interfaces.json').read_text())
m=mujoco.MjModel.from_xml_path(str(a.bundle/'model/robot.xml'))
# v2 used the upstream property's misleading world-frame description. Its
# implementation actually exposes Newton model.body_inertia (body-local).
native=np.array(raw.get('inertia_body_kg_m2',raw.get('inertia_world_kg_m2'))).reshape(-1,3,3)
checks=[]
for index,link in enumerate(iface['links']):
 i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,link['name'])
 r=np.empty(9);mujoco.mju_quat2Mat(r,m.body_iquat[i]);r=r.reshape(3,3)
 expected=r@np.diag(m.body_inertia[i])@r.T
 error=float(np.max(np.abs(native[index]-expected)))
 checks.append(dict(body=link['name'],status='pass' if np.allclose(native[index],expected,atol=1e-9,rtol=2e-4) else 'fail',max_error_kg_m2=error,expected_body_tensor=expected.tolist(),native_body_tensor=native[index].tolist()))
for path,name in [(a.loaded,'loaded.json'),(a.bundle/'manifest.json','bundle-manifest.json'),(Path(__file__),'audit_native_inertia.py')]:snapshot_file(path,a.output/'source'/name)
report=dict(status='pass' if all(x['status']=='pass' for x in checks) else 'fail',frame='body-local about center of mass',legacy_field_name='inertia_world_kg_m2 is a historical mislabel; original evidence unchanged',max_error_kg_m2=max(x['max_error_kg_m2'] for x in checks),scope='Native loaded inertia transfer only; no measured hardware or dynamic identification',checks=checks)
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(report['status'],report['max_error_kg_m2'])
raise SystemExit(report['status']!='pass')
