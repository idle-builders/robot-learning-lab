"""Exercise the extracted package through native and actual MJLab consumers."""
import os,subprocess,tempfile,zipfile
from common import *

def main():
    package=OUT/f'cove50-description-{VERSION}.zip'
    with tempfile.TemporaryDirectory(prefix='cove50-portable-')as folder:
        with zipfile.ZipFile(package)as z:z.extractall(folder)
        description=Path(folder)/'cove50_description'
        env=dict(os.environ,COVE50_MODEL_DIR=str(description),PYTHONPATH=str(description/'runtime'))
        code='''import json,mujoco,numpy as np
from common import OUT,read
from runtime import Runtime,metric_point
rows={}
for name in ['cove50.urdf','cove50-rigid.xml','cove50-compliant.xml']:
 m=mujoco.MjModel.from_xml_path(str(OUT/name));rows[name]=dict(nq=m.nq,nv=m.nv,nu=m.nu,mass_kg=float(m.body_mass.sum()))
m=mujoco.MjModel.from_xml_path(str(OUT/'cove50-rigid.xml'));d=mujoco.MjData(m);d.qpos[2]=.3022;mujoco.mj_forward(m,d);r=Runtime(m,d)
r.command(r.names,np.zeros(22))
for _ in range(30):r.update();mujoco.mj_step(m,d)
o=r.observation()
print(json.dumps(dict(models=rows,runtime=dict(named_actions=len(r.names),observation_values=len(o['values']),feedback_age_s=o['age_s'],finite=bool(np.isfinite(o['values']).all())),package_root=str(OUT))))
'''
        native=subprocess.run([str(ROOT/'.venv/bin/python'),'-c',code],cwd=folder,env=env,text=True,capture_output=True,check=True)
        code='''import json
from common import OUT
from mjlab_adapter import robot_cfg
from mjlab.entity import Entity
e=Entity(robot_cfg());m=e.compile()
print(json.dumps(dict(nq=m.nq,nv=m.nv,nu=m.nu,mass_kg=float(m.body_mass.sum()),actuator_names=list(e.actuator_names))))
'''
        mjlab=subprocess.run([str(ROOT/'.venv-control/bin/python'),'-c',code],cwd=folder,env=env,text=True,capture_output=True,check=True)
        record=dict(schema='cove50.portable-consumer.v1',package=ref(package),native=json.loads(native.stdout.splitlines()[-1]),
            mjlab=json.loads(mjlab.stdout.splitlines()[-1]),native_stderr=native.stderr,mjlab_stderr=mjlab.stderr,
            scope='Fresh temporary extraction, actual native URDF/MJCF and Runtime, actual MJLab Entity from extracted runtime. Repository PYTHONPATH absent, package directory explicit. No ROS installation or GPU rollout claimed.',
            cleanup='Temporary extraction removed after verified consuming calls; authoritative archive remains.')
        record['native'].pop('package_root',None)
        write(OUT/'portable-consumer-verification.json',record);print(record)

if __name__=='__main__':main()
