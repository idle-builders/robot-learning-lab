"""Shared-tree morphology adaptation, retaining unchanged CAD and mass evidence."""
import copy,json,sys
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lab.config import load_config
from lab.robot import make_robot
from lab.export import mjcf
from lab.artifact_store import snapshot_file,retain_dependency

def build(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 robot=make_robot(load_config(ROOT/'variants/pip-floor-reach-v1.yaml'))
 mjcf(robot,out/'generated.xml')
 generated=ET.parse(out/'generated.xml').getroot()
 bundle=ROOT/'artifacts/pip-cad-gym-bundle-v4'
 root=ET.parse(bundle/'model/robot.xml').getroot()
 for mesh in root.findall('./asset/mesh'):mesh.set('file',str((bundle/'model'/mesh.get('file')).absolute()))
 changed=[]
 for side in ['left','right']:
  name=side+'_upper_arm';body=root.find(f".//body[@name='{name}']");fresh=generated.find(f".//body[@name='{name}']")
  # Changed link uses freshly generated envelope and inertia. Old detailed CAD
  # and full-link mass are not silently stretched or claimed to match.
  for node in list(body):
   if node.tag in ['geom','inertial']:body.remove(node)
  for node in fresh:
   if node.tag in ['geom','inertial']:
    node=copy.deepcopy(node)
    if node.tag=='geom':node.set('rgba','.55 .45 .7 1')
    body.append(node)
  name=side+'_forearm';body=root.find(f".//body[@name='{name}']");fresh=generated.find(f".//body[@name='{name}']")
  body.set('pos',fresh.get('pos'));changed.append(side+'_upper_arm')
 for actuator in root.findall('./actuator/motor'):
  if actuator.get('name').endswith(('_thigh_joint_motor','_shin_joint_motor')):actuator.set('ctrlrange','-1 1')
 ET.ElementTree(root).write(out/'robot.xml')
 (out/'resolved-config.json').write_text(robot.config.model_dump_json(indent=2))
 (out/'interfaces.json').write_text(json.dumps(robot.interface(),indent=2))
 (out/'adaptation.json').write_text(json.dumps({'changed_dimensions':{'upper_arm_m':{'from':.085,'to':.100}},'changed_links':changed,'leg_effort_cap_nm':1.,'evidence':'unmeasured_design_assumption','unchanged_links':'retained sealed CAD and planning mass; changed upper arms have generic envelope/inertia, not production mechanisms','source_bundle':str(bundle.relative_to(ROOT))},indent=2))
 for rel in ['variants/pip-floor-reach-v1.yaml','lab/robot.py','lab/export.py','tools/build_floor_reach_model.py','robot/base.yaml']:snapshot_file(ROOT/rel,out/'source'/rel)
 (out/'dependency.json').write_text(json.dumps(retain_dependency(bundle),indent=2))
if __name__=='__main__':build(sys.argv[1])
