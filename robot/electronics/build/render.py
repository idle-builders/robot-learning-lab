"""Opaque shell context and separately labeled native-CAD inspection views."""
from pathlib import Path
import hashlib,json,struct,subprocess
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).parent
BASE=ROOT/'artifacts/cove50-electronics-integration/r1-20261026'
SOURCE=BASE;OUT=SOURCE/'renders';OUT.mkdir(exist_ok=True)
scene=trimesh.load(SOURCE/'inspection.glb',force='scene')
exe=Path('/tmp/cove-power-internal-zbuffer');renderer=ROOT/'robot/electronics/power/cad/zbuffer_renderer.cpp'
subprocess.run(['clang++','-std=c++17','-O3',str(renderer),'-o',str(exe)],check=True)
geometry=[]
for node in scene.graph.nodes_geometry:
 T,key=scene.graph[node];mesh=scene.geometry[key].copy();mesh.apply_transform(T);mesh.apply_scale(1000);mesh.process(validate=True)
 cap=len(mesh.faces) if node.startswith('ape_body_shell__')else 18000 if node in ['sensor_head_integrated_v17','back_touch_cap_v17']else 4500
 if node in ['ape_body_shell__torso_front_internal_lightwell_r18','ape_body_shell__torso_back','ape_body_shell__pelvis_front','ape_body_shell__pelvis_back']:cap=len(mesh.faces)
 if len(mesh.faces)>cap:
  original=mesh;reduced=mesh.simplify_quadric_decimation(face_count=cap)
  if np.max(abs(reduced.bounds-original.bounds))<=.1:mesh=reduced
 color=np.array([.60,.63,.67])
 if node.startswith(('Power_supervisor','Main_precharge')):color=np.array([.12,.43,.32])
 if 'tray'in node or 'clamp_'in node:color=np.array([.51,.36,.65])
 if node.startswith('Common_power'):color=np.array([.27,.53,.57])
 if node.startswith(('Main_frame_','Supervisor_frame_','Cell7_')):color=np.array([.70,.70,.67])
 if node.startswith('Main_frame_')and'spacer'in node:color=np.array([.75,.67,.53])
 if node.startswith('ape_body_shell__'):
  color=np.array([.88,.86,.79]);low=node.lower()
  if any(x in low for x in ['flex_envelope','cuff','collar','trim','sole','boot','paw']):color=np.array([.43,.39,.34])
  if 'face_mask'in low:color=np.array([.69,.56,.42])
  if 'muzzle'in low:color=np.array([.84,.74,.59])
  if any(x in low for x in ['dark_eye','nostril','mouth_inlay']):color=np.array([.10,.09,.08])
  if 'indicator'in low:color=np.array([.96,.43,.10])
 if node.startswith('Cell_'):
  color=np.array([.82,.56,.23])
  if 'PCB'in node:color=np.array([.18,.48,.31])
  if 'cover'in node:color=np.array([.57,.37,.68])
  if 'mPPE'in node:color=np.array([.20,.22,.24])
  if 'Cu'in node or 'bundle'in node:color=np.array([.76,.44,.21])
  if '5413'in node:color=np.array([.74,.56,.25])
 if node.startswith('Common_frame'):color=np.array([.27,.53,.57])
 if 'Pico2'in node or 'Radxa_ZERO'in node:color=np.array([.18,.43,.31])
 if 'source_family'in node:color=np.array([.28,.56,.72])
 if node in ['body_IMU_4517','foot_ADC_6494']:color=np.array([.68,.28,.22])
 if node=='BMS_unrouted_package_reservation':color=np.array([.62,.35,.69])
 if node=='sensor_head_integrated_v17':color=np.array([.88,.86,.79])
 if node=='back_touch_cap_v17':color=np.array([.92,.59,.18])
 if node.startswith('lower_cradle_lugs'):color=np.array([.55,.39,.68])
 if node in ['lower_cradle_lugs_224_with_waist_guard_r18','lower_cradle_lugs_252_retained_outer_r18']:color=np.array([.49,.51,.52])
 if node.startswith(('lower_cradle_screw','lower_cradle_nut')):color=np.array([.58,.61,.64])
 if node.startswith('BMS_'):
  color=np.array([.59,.44,.69])
  if 'board' in node:color=np.array([.15,.42,.24])
  if 'bundle' in node:color=np.array([.76,.44,.21])
  if 'mPPE' in node:color=np.array([.14,.16,.18])
  if 'solder' in node:color=np.array([.63,.65,.66])
  if 'screw' in node:color=np.array([.76,.74,.66])
 if node.startswith(('B0_RAW__','B1__','B2__','B3_RAW__')):color=np.array([.61,.40,.74])
 if node.startswith(('B0_RAW__','B1__','B2__','B3_RAW__'))and'C11000'in node:color=np.array([.78,.48,.20])
 if node.startswith('Cell_')and'_positive__mPPE'in node:color=np.array([.79,.22,.15])
 if node.startswith('Cell_')and'_negative__mPPE'in node:color=np.array([.18,.21,.24])
 if node.startswith('Cell_capture_bridge'):color=np.array([.51,.36,.65])
 if node.startswith('Cell_capture_')and not node.startswith('Cell_capture_bridge'):color=np.array([.65,.66,.64])
 if node=='original_profile_head_skin_v23':color=np.array([.88,.86,.79])
 if node in ['ROCK4D_official','B0410_supplier_native_v23','B0196_supplier_native_v23','Pololu_D24V50F5_official_alternative']:color=np.array([.17,.44,.31])
 if node in ['single_shared_carrier_v23','combined_RF_and_regulator_bridge']:color=np.array([.52,.37,.66])
 if node=='Radian_INL25001_bare':color=np.array([.42,.44,.47])
 geometry.append((node,mesh,color))
def render(view,opened,tag,region=None,actual_head=False,only_cell=None,lower_detail=False,bms_detail=False,hide_guard=False):
 tri=[];colors=[]
 for name,m,c in geometry:
  if bms_detail and not(name.startswith('BMS_')or name.startswith('Common_frame')):continue
  if hide_guard and name=='BMS_recessed_fastener_front_guard':continue
  if only_cell is not None and not name.startswith(only_cell):continue
  if lower_detail and not(name.startswith(('Cell_03__','Cell_04__','Cell_05__','lower_cradle_'))or name=='Common_frame_lower_hangers_r9'):continue
  if name=='sensor_head_integrated_v17'and not actual_head:continue
  if actual_head and name in ['ape_body_shell__head_cranium_front','ape_body_shell__head_cranium_back']:continue
  if opened and name=='original_profile_head_skin_v23':continue
  if opened and name.startswith('ape_body_shell__')and any(v in name for v in ['torso_front','torso_back','pelvis_front','pelvis_back','chest_indicator']):continue
  light=np.array([.3,-.5,.8]);shade=.62+.38*np.abs(m.face_normals@light);tri.append(m.triangles);colors.append(c*shade[:,None])
 tri=np.concatenate(tri);colors=np.concatenate(colors);view=np.array(view,float);view/=np.linalg.norm(view);right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
 xy=np.stack([tri@right,tri@up],axis=-1);W,H=1000,1500
 if region:
  lo3,hi3=np.array(region);corners=np.array([[x,y,z]for x in [lo3[0],hi3[0]]for y in [lo3[1],hi3[1]]for z in [lo3[2],hi3[2]]]);projected=np.stack([corners@right,corners@up],axis=-1);lo=projected.min(0);hi=projected.max(0)
 else:lo=xy.min(axis=(0,1));hi=xy.max(axis=(0,1))
 scale=min(W/(hi[0]-lo[0]),H/(hi[1]-lo[1]))*.91;mid=(lo+hi)/2
 xyz=np.empty_like(tri);xyz[:,:,0]=(xy[:,:,0]-mid[0])*scale+W/2;xyz[:,:,1]=H/2-(xy[:,:,1]-mid[1])*scale;xyz[:,:,2]=tri@view
 data=np.concatenate([xyz.reshape(-1,9),colors],axis=1).astype('<f4');src=Path(f'/tmp/cove-internal-r17-{tag}.bin');dst=src.with_suffix('.ppm')
 subprocess.run([str(exe),'/dev/stdin',str(dst)],input=struct.pack('<III',W,H,len(tri))+data.tobytes(),check=True);im=Image.open(dst).copy();dst.unlink();return im

for tag,opened,region in [('complete-ape',False,None),('electronics-cutaway',True,[[-85,-55,180],[85,60,505]])]:
 fig,axes=plt.subplots(1,3,figsize=(13,8),facecolor='#f6f5f1');fig.subplots_adjust(left=.01,right=.99,bottom=.13,top=.88,wspace=.02)
 for i,(ax,view,title)in enumerate(zip(axes,[[.25,-1,.08],[.55,1,.08],[1,0,.03]],['Front','Rear','Right side'])):
  ax.imshow(render(view,opened,tag+str(i),region));ax.axis('off');ax.set_title(title,fontsize=12)
 fig.suptitle('Selected electronics composition · sensing V24 / compute R23 / power R33',fontsize=14)
 fig.text(.5,.075,'853 source parts · original compact head · 3S3P battery and shared torso carrier · selected motor geometry',ha='center',fontsize=9)
 fig.text(.5,.043,'Static CAD composition. Local 4mm torso relief selected. Remaining supply/data harness and physical qualification are open.',ha='center',fontsize=8,color='#555555')
 fig.savefig(OUT/(tag+'.png'),dpi=160);plt.close(fig);print(tag,flush=True)
