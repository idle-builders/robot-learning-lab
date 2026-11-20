"""Source-bound linear tetra elasticity API. No import-time loaded solve."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
ROOT=Path(__file__).resolve().parents[3]
CAD=ROOT/'artifacts/cove50-engineering-competition/waist-pelvis-20261021/B/compact-shin-v5-r1/final'
OUT=ROOT/'artifacts/cove50-engineering-competition/waist-pelvis-20261021/B/compact-shin-fem-r1'
CAD_FREEZE='0c00f072529dc6fcf2207207875c523132825a099546f297b3c96325300aa3f7'
CAD_PARTS='4069cb398155ae658198f3ae4cb606d1e924766011cddee488e251f5e015afc7'
ASSEMBLER=ROOT/'robot/body/loads/rs00-cup-elasticity/solve.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def source_part(kind):
 if sha(CAD/'frozen-snapshot.json')!=CAD_FREEZE or sha(CAD/'parts.json')!=CAD_PARTS:raise RuntimeError('CAD input binding changed; require a new explicit source version')
 name={'shin':'R_B_centred_compact_shin_v5_r1','left_coupler':'L_B_handed_direct_horn_coupler_v5_r4'}[kind]
 row=next(p for p in json.loads((CAD/'parts.json').read_text())['parts']if p['name']==name)
 if sha(ROOT/row['native_step'])!=row['native_step_sha256']:raise RuntimeError('Native input hash mismatch')
 return row

def reference_namespace(D=None):
 source=ASSEMBLER.read_text();tree=ast.parse(source);selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in['elasticity_matrix','assemble','von_mises']],type_ignores=[])
 ns={'np':np,'coo_matrix':coo_matrix,'E':68300.,'NU':.33};exec(compile(selected,str(ASSEMBLER),'exec'),ns)
 if D is not None:ns['elasticity_matrix']=lambda:np.asarray(D,float)
 return ns

def isotropic(E,nu):return reference_namespace()['elasticity_matrix'](float(E),float(nu))
def assemble_materials(points,tet,material_matrices,cell_material_ids=None):
 ids=np.zeros(len(tet),np.int32)if cell_material_ids is None else np.asarray(cell_material_ids,np.int32)
 K=csr_matrix((3*len(points),)*2);vol=np.empty(len(tet));B=np.empty((len(tet),6,12))
 for key in np.unique(ids):
  mask=ids==key;kk,v,bb,_=reference_namespace(material_matrices[int(key)])['assemble'](points,tet[mask]);K+=kk;vol[mask]=v;B[mask]=bb
 return K,vol,B,ids

def wrench(points,forces,origin):
 f=np.asarray(forces).reshape(-1,3);return np.r_[f.sum(axis=0),np.cross(points-np.asarray(origin),f).sum(axis=0)]
def minimum_weight_wrench(points,nodal_area,target,origin):
 nodes=np.flatnonzero(nodal_area>0);A=np.zeros((6,3*len(nodes)))
 for i,node in enumerate(nodes):
  x,y,z=points[node]-np.asarray(origin);A[:3,3*i:3*i+3]=np.eye(3);A[3:,3*i:3*i+3]=[[0,-z,y],[z,0,-x],[-y,x,0]]
 weights=np.repeat(nodal_area[nodes],3);normal=(A*weights)@A.T;u,s,v=np.linalg.svd(normal);result=np.zeros((len(points),3))
 result[nodes]=(weights*(A.T@np.linalg.solve(normal,np.asarray(target,float)))).reshape(-1,3)
 return result,dict(nodes=len(nodes),normal_singular_values=s.tolist(),target_wrench_N_Nmm=list(target),actual_wrench_N_Nmm=wrench(points,result,origin).tolist())

def groups(kind,points,tri):
 vertices=points[tri];centres=vertices.mean(axis=1);area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
 masks={}
 def plane(axis,value):return np.all(abs(vertices[:,:,axis]-value)<2e-6,axis=1)
 def disks(axes,centers,radius):return np.logical_or.reduce([np.sum((centres[:,axes]-np.asarray(c))**2,axis=1)<=radius**2 for c in centers])
 if kind=='shin':
  yz=[[4,116],[4,130]];xz=[[62.7,51],[62.7,75],[74.7,51],[74.7,75]]
  masks['support_front_annuli']=plane(0,41.275)&disks([1,2],yz,3.)
  masks['support_rear_annuli']=plane(0,78.725)&disks([1,2],yz,3.)
  masks['load_four_case_annuli']=plane(1,18.25)&disks([0,2],xz,2.25)
  origin=np.array([66.,4.,47.]);expected=dict(support_front_annuli=2*np.pi*(3**2-1.6**2),support_rear_annuli=2*np.pi*(3**2-1.6**2),load_four_case_annuli=4*np.pi*(2.25**2-1.35**2))
 else:
  yz=[[4,39],[4,55],[-4,47],[12,47]];xz=[[-69,19],[-81,19],[-75,13],[-75,25]]
  masks['support_upper_horn_annuli']=plane(0,-87.7)&disks([1,2],yz,2.)
  masks['load_lower_horn_annuli']=plane(1,-32)&disks([0,2],xz,2.)
  origin=np.array([-75.,-13.5,19.]);expected={k:4*np.pi*(2**2-1.1**2)for k in masks}
 weights={k:np.bincount(tri[m].ravel(),weights=np.repeat(area[m]/3,3),minlength=len(points))for k,m in masks.items()}
 summary={k:dict(triangles=int(m.sum()),nodes=int(np.count_nonzero(weights[k])),area_mm2=float(area[m].sum()),analytic_untrimmed_annulus_area_mm2=expected[k],relative_area_difference=float(area[m].sum()/expected[k]-1))for k,m in masks.items()}
 return masks,weights,summary,origin
