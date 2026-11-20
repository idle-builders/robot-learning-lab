"""Linear-tetra elastic unit-response model of the reviewed +Y RS00 cup.

Units: mm, N, MPa. The boundary springs are declared ideal bilateral
constraints, not a fitted bearing or motor-contact law.
"""
import json
import sys
from pathlib import Path

import meshio
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from boundary_board import patches

HERE = Path(__file__).resolve().parent
AXIS = np.array([-4.0, 0.0, 275.0])
E = 68300.0  # N/mm²; NASA-hosted 6061-T6 sheet data, PDF p36
NU = .33


def elasticity_matrix(E=E, nu=NU):
    lam = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    D = np.zeros((6,6))
    D[:3,:3] = lam
    np.fill_diagonal(D[:3,:3], lam+2*mu)
    D[3,3]=D[4,4]=D[5,5]=mu
    return D


def assemble(points, tet):
    n = len(points)
    D = elasticity_matrix()
    rows=[]; cols=[]; data=[]
    volumes=np.empty(len(tet))
    Bs=np.empty((len(tet),6,12))
    for start in range(0,len(tet),1500):
        stop=min(start+1500,len(tet))
        cells=tet[start:stop]
        xyz=points[cells]
        A=np.concatenate((np.ones((len(cells),4,1)),xyz),axis=2)
        inv=np.linalg.inv(A)
        grad=np.transpose(inv[:,1:,:],(0,2,1))
        vol=np.abs(np.linalg.det(A))/6
        if np.min(vol) <= 1e-12:
            raise ValueError('near-degenerate element')
        B=np.zeros((len(cells),6,12))
        for i in range(4):
            g=grad[:,i,:]
            a=3*i
            B[:,0,a]=g[:,0]; B[:,1,a+1]=g[:,1]; B[:,2,a+2]=g[:,2]
            B[:,3,a]=g[:,1]; B[:,3,a+1]=g[:,0]
            B[:,4,a+1]=g[:,2]; B[:,4,a+2]=g[:,1]
            B[:,5,a]=g[:,2]; B[:,5,a+2]=g[:,0]
        Ke=np.einsum('eai,ab,ebj,e->eij',B,D,B,vol,optimize=True)
        dof=3*cells[:,:,None]+np.arange(3)[None,None,:]
        dof=dof.reshape((-1,12))
        rows.append(np.broadcast_to(dof[:,:,None],Ke.shape).reshape(-1).copy())
        cols.append(np.broadcast_to(dof[:,None,:],Ke.shape).reshape(-1).copy())
        data.append(Ke.reshape(-1))
        volumes[start:stop]=vol
        Bs[start:stop]=B
    K=coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(3*n,3*n)).tocsr()
    K.sum_duplicates()
    return K,volumes,Bs,D


def boundary_data(mesh):
    points=mesh.points[:,:3]
    tri=np.vstack([block.data for block in mesh.cells if block.type=='triangle'])
    p=points[tri]
    centers,area,groups=patches(points,tri)
    if min(int(mask.sum()) for mask in groups.values()) < 10:
        raise ValueError('undersampled physical patch')
    return points,tri,centers,area,groups


def spring_matrix(points,tri,area,groups,k):
    matrices={}
    summaries={}
    for group in ('bearing_radial','bearing_axial_shoulder','motor_bolt_tangential'):
        mask=groups[group]
        nodal=np.bincount(tri[mask].ravel(),weights=np.repeat(area[mask]/3,3),minlength=len(points))
        nodes=np.flatnonzero(nodal>0)
        d=np.zeros((len(nodes),3))
        if group=='bearing_axial_shoulder':
            d[:,1]=1
        else:
            radial=points[nodes][:,(0,2)]-AXIS[[0,2]]
            radial/=np.linalg.norm(radial,axis=1)[:,None]
            if group=='bearing_radial':
                d[:,0]=radial[:,0];d[:,2]=radial[:,1]
            else:
                d[:,0]=radial[:,1];d[:,2]=-radial[:,0]
        # Lumped surface Winkler penalty ∫k(u·d)^2 dA, in N/mm³.
        block=k*nodal[nodes,None,None]*d[:,:,None]*d[:,None,:]
        dof=3*nodes[:,None]+np.arange(3)[None,:]
        rows=np.broadcast_to(dof[:,:,None],block.shape).reshape(-1).copy()
        cols=np.broadcast_to(dof[:,None,:],block.shape).reshape(-1).copy()
        matrices[group]=coo_matrix((block.ravel(),(rows,cols)),shape=(3*len(points),)*2).tocsr()
        matrices[group].sum_duplicates()
        summaries[group]={'nodes':len(nodes),'lumped_area_mm2':float(nodal[nodes].sum())}
    K=sum(matrices.values(),csr_matrix((3*len(points),)*2))
    return K,summaries,matrices


def load_vector(points,tri,area,groups,case):
    mask=groups['distal_flange_load']
    nodal=np.bincount(tri[mask].ravel(),weights=np.repeat(area[mask]/3,3),minlength=len(points))
    nodes=np.flatnonzero(nodal>0)
    f=np.zeros((len(points),3))
    if case=='unit_x_force':
        f[nodes,0]=nodal[nodes]/nodal[nodes].sum()
    elif case=='unit_pitch_moment':
        # Minimum weighted nodal force norm subject to exact zero resultant
        # and an exact +Y moment about AXIS, including the other two moments.
        A=np.zeros((6,3*len(nodes)))
        for i,node in enumerate(nodes):
            x,y,z=points[node]-AXIS
            A[:3,3*i:3*i+3]=np.eye(3)
            A[3:,3*i:3*i+3]=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        weights=np.repeat(nodal[nodes],3)
        target=np.array([0.,0.,0.,0.,1.,0.])
        f[nodes]=(weights*(A.T@np.linalg.solve((A*weights)@A.T,target))).reshape((-1,3))
    else:
        raise ValueError(case)
    return f.reshape(-1),nodal


def wrench(points,forces,origin=AXIS):
    v=forces.reshape((-1,3))
    return {'force_N':v.sum(axis=0).tolist(),
            'moment_Nmm':np.cross(points-origin,v).sum(axis=0).tolist()}


def von_mises(stress):
    sx,sy,sz,txy,tyz,txz=stress.T
    return np.sqrt(((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)/2+3*(txy*txy+tyz*tyz+txz*txz))


def run(size, k, case):
    mesh=meshio.read(HERE/f'cup-h{size:g}.msh')
    points,tri,centers,area,groups=boundary_data(mesh)
    tet=np.vstack([block.data for block in mesh.cells if block.type=='tetra'])
    K,vol,B,D=assemble(points,tet)
    Ks,support,support_matrices=spring_matrix(points,tri,area,groups,k)
    f,load_weights=load_vector(points,tri,area,groups,case)
    u=spsolve(K+Ks,f)
    if not np.all(np.isfinite(u)):
        raise ValueError('nonfinite solution')
    reaction=-Ks@u
    by_patch={name:wrench(points,-matrix@u) for name,matrix in support_matrices.items()}
    for direction in ('force_N','moment_Nmm'):
        summed=np.sum([entry[direction] for entry in by_patch.values()],axis=0)
        if not np.allclose(summed,wrench(points,reaction)[direction],rtol=0,atol=1e-6):
            raise ValueError('support patch reaction decomposition failed')
    total=wrench(points,f+reaction)
    if np.linalg.norm(total['force_N'])>1e-5 or np.linalg.norm(total['moment_Nmm'])>1e-3:
        raise ValueError('reaction equilibrium failed: '+str(total))
    external=np.dot(f,u)
    material=np.dot(u,K@u)/2
    springs=np.dot(u,Ks@u)/2
    if abs(external-2*(material+springs))>1e-6*max(1,abs(external)):
        raise ValueError('energy balance failed')
    uc=u.reshape((-1,3))[tet].reshape((-1,12))
    strain=np.einsum('eij,ej->ei',B,uc)
    stress=strain@D.T
    vm=von_mises(stress)
    tc=points[tet].mean(axis=1)
    regions={'bearing_seat':(tc[:,2]>=239)&(tc[:,2]<=311)&(tc[:,1]>=19)&(tc[:,1]<=28),
             'thin_sleeve':(tc[:,2]>=239)&(tc[:,2]<=311)&(tc[:,1]>=31)&(tc[:,1]<=51),
             'taper_web':(tc[:,2]>=240)&(tc[:,2]<=311)&(tc[:,1]>=55),
             'descending_spar':(tc[:,2]>=157)&(tc[:,2]<=230),
             'leg_flange':(tc[:,2]>=132)&(tc[:,2]<=148)}
    region_stats={name:{'elements':int(m.sum()),'vm_p50_MPa':float(np.quantile(vm[m],.5)),
                        'vm_p95_MPa':float(np.quantile(vm[m],.95)),
                        'vm_max_MPa_singular_not_strength':float(np.max(vm[m]))}
                  for name,m in regions.items() if m.any()}
    displacement=u.reshape((-1,3))
    if size == 2 and k == 10000 and case == 'unit_x_force':
        np.savez_compressed(HERE/'field-h2-unit-x.npz', points=points, tet=tet,
                            displacement_mm=displacement, cell_centers_mm=tc,
                            von_mises_MPa=vm)
    load_nodes=np.flatnonzero(load_weights>0)
    result={'mesh_size_mm':size,'spring_modulus_N_mm3':k,'case':case,
            'nodes':len(points),'tets':len(tet),'mesh_volume_mm3':float(vol.sum()),
            'material_E_MPa':E,'poisson':NU,'patches':support,
            'applied':wrench(points,f),'reaction':wrench(points,reaction),
            'reaction_by_patch':by_patch,
            'force_moment_residual':total,
            'flange_mean_displacement_mm':np.average(displacement[load_nodes],axis=0,weights=load_weights[load_nodes]).tolist(),
            'max_displacement_mm':float(np.linalg.norm(displacement,axis=1).max()),
            'work_Nmm':float(external),'strain_energy_Nmm':float(material),
            'support_spring_energy_Nmm':float(springs),
            'regions':region_stats}
    return result


if __name__=='__main__':
    result=run(float(sys.argv[1]),float(sys.argv[2]),sys.argv[3])
    print(json.dumps(result,indent=2))
    (HERE/f"solve-h{sys.argv[1]}-k{sys.argv[2]}-{sys.argv[3]}.json").write_text(json.dumps(result,indent=2)+'\n')
