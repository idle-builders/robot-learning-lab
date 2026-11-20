"""Transfer retained, geometry-bound FEM compliance, preserving coupled terms."""
from common import *

REVIEW = ROOT/'artifacts/cove50-strength/connector-review-20261024/connector-review.json'

def main():
    manifest = read(CAD/'assembly-parts.json')
    parts = {p.get('source_name', p['name']):p for p in manifest['parts']}
    joints = {j['name']:j for j in manifest['joints']}
    blocks, inventory = [], []
    seen = set()
    for record in read(REVIEW)['parts']:
        name, area, side = record['part'], record['area'], record.get('side')
        present = name in parts
        row = dict(part=name, area=area, present_in_selected_CAD=present, source_report=record['baseline_report'])
        inventory.append(row)
        if not present:
            row['disposition']='No transfer: source part replaced or absent in composed R33/R23 assembly'
            continue
        rp=ROOT/record['baseline_report']['path']; report=read(rp)
        key=(str(rp),side)
        if key in seen:
            row['disposition']='Shared coupled fixture counted once, not one spring per component'
            continue
        seen.add(key)
        receiving_case = None
        if area in ['hip-yaw','hip-pink','knee-green','shin','foot-brackets']:
            bp=ROOT/report['basis']['path']
            with np.load(bp) as z: c=z['reference_dofs_mm_rad'][:,:6].copy()
            mp=ROOT/report['mesh_report']['path']; meta=read(mp)
            origin=np.array(meta['fixture']['load_reference_origin_mm'],float)
            suffix={'hip-yaw':'hip_pitch','hip-pink':'hip_roll','knee-green':'knee','shin':'ankle_pitch','foot-brackets':'ankle_roll'}[area]
            target=side+'_'+suffix
            candidates={'hip-yaw':f'lower__{side}_pitch_servo_case_reference',
                'hip-pink':f'lower__{side}_roll_servo_case_reference',
                'knee-green':f'lower__{side}_C047_knee_case',
                'shin':f'lower__{side}_B13R14_raised_C047_ankle_pitch_case'}
            receiving_case=candidates.get(area)
            source_parts=report['source_parts']
            transfer='Direct six-DOF rigid receiving interface in retained source FEM'
            placement='after' if area=='foot-brackets' else 'before'
            # Every member in a two-bracket fixture is one coupled solve.
            matched=[]
            for sp in source_parts:
                hits=[p for p in manifest['parts'] if p.get('native_step')==sp['path'] or p.get('native_step_sha256')==sp['sha256']]
                matched.append(dict(source=sp, exact_selected_hits=[p['name']for p in hits]))
            row['geometry_identity']=matched
            if not all(x['exact_selected_hits']for x in matched):
                raise ValueError('FEM part identity needs reconciliation: '+name)
        elif area.startswith('upper-F'):
            basis_value=report.get('unit_basis')
            bp=ROOT/basis_value if isinstance(basis_value,str) else rp.parent/'unit-basis.npz'
            if not bp.is_file():
                row['disposition']='No retained reduced basis found';continue
            source=read(ROOT/report['source_unit_report']) if report.get('source_unit_report') else report
            with np.load(bp) as z: c=z['load_rigid_displacement'][:,:6].copy()
            origin=np.array(source['reference_origin_mm'],float)
            # Left *manufactured sheet* reflection is documented by the source;
            # polar translation/force and axial rotation/moment differ in sign.
            if side=='L':
                mirror=np.diag([-1.,1.,1.]); H=np.zeros((6,6));H[:3,:3]=mirror;H[3:,3:]=-mirror
                c=H@c@H.T;origin=mirror@origin
            suffix={'upper-F1':'shoulder_roll','upper-F2':'shoulder_yaw','upper-F3':'elbow','upper-F4':'wrist_fixed'}[area]
            target=side+'_'+suffix
            receiving_case={'upper-F1':f'upper__{side}_roll_C047_case','upper-F2':f'upper__{side}_yaw_C047_case',
                'upper-F3':f'upper__{side}_elbow_C047_main_horn'}.get(area)
            source_parts=[dict(path=source['native_step'],sha256=source['native_step_sha256'])]
            placement='before'
            transfer='Retained reduced six-DOF receiving patch; source-published reflected manufactured left sheet mapping'
            # F4 is retained as a calibrated interface sidecar. The selected
            # source has no separate physical hand body or its mass split.
            if area=='upper-F4': placement='sidecar_only'
        else:
            row['disposition']='Distributed continuum/source fixture retained; no justified six-DOF rigid receiver reduction available'
            continue
        # q=[mm,rad], w=[N,Nmm] -> q=[m,rad], w=[N,Nm].
        c=np.diag([.001]*3+[1.]*3)@c@np.diag([1.]*3+[1000.]*3)
        rotation=np.zeros((6,6));rotation[:3,:3]=R;rotation[3:,3:]=R
        c=rotation@c@rotation.T
        raw_c=c.copy()
        symmetry=float(np.linalg.norm(c-c.T)/np.linalg.norm(c))
        c=(c+c.T)/2
        original_origin=R@(origin/1000)
        target_origin=R@(np.array(joints[target]['origin_world_mm'])/1000) if target in joints else original_origin
        delta=target_origin-original_origin
        B=np.eye(6);B[:3,3:]=-skew(delta)
        c=B@c@B.T
        # Scale angular coordinates with a declared 50mm length only for
        # eigendecomposition/conditioning; this does not change elastic energy.
        L=.05;D=np.diag([1.]*3+[L]*3)
        scaled=D@c@D
        eigen=np.linalg.eigvalsh(scaled)
        if eigen.min() <= 0: raise ValueError('Non-positive source compliance: '+name)
        k=np.linalg.inv(c)
        ks=np.linalg.inv(D)@k@np.linalg.inv(D)
        lam,v=np.linalg.eigh((ks+ks.T)/2)
        coefficients=(D@v).T
        reconstruction=coefficients.T@np.diag(lam)@coefficients
        block=dict(id=(side+'-'if side else '')+area, source_parts=source_parts,
            receiving_joint=target, placement=placement, moved_parent_part=receiving_case,
            source_origin_CAD_mm=origin.tolist(), model_reference_origin_m=target_origin.tolist(),
            reference_shift_model_m=delta.tolist(), compliance_SI=c.tolist(), stiffness_SI=k.tolist(),
            raw_compliance_at_source_origin_SI=raw_c.tolist(),
            coordinates=['tx_m','ty_m','tz_m','rx_rad','ry_rad','rz_rad'],
            wrench=['Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm'],
            source_reciprocity_relative_error=symmetry, scaled_compliance_eigenvalues=eigen.tolist(),
            characteristic_length_m=L, modal_stiffness_N_m=lam.tolist(), modal_coefficients=coefficients.tolist(),
            modal_reconstruction_relative_error=float(np.linalg.norm(reconstruction-k)/np.linalg.norm(k)),
            numerical_model='6 infinitesimal passive coordinates with six coupled fixed-tendon modal springs; exact source tangent K at neutral, finite rotation is only a reduced approximation',
            damping='Not identified by static FEM. Native scenario uses modal critical fraction0.10 against assumed0.055kg modal reference; sweep0.02..0.20. Not fabricated-part damping.',
            source_refs=[ref(rp),ref(bp)]+([ref(mp)] if area in ['hip-yaw','hip-pink','knee-green','shin','foot-brackets'] else [ref(ROOT/source['native_step'])]),
            transfer_basis=transfer,
            limits=['Ideal clamps and rigid receiving patches; no measured preload/slip/gap/gearbox/bearing/tap compliance',
                    'Small-displacement isotropic linear material tangent; severe source fields exceed elastic applicability',
                    'Source ideal selfweight distributed field is retained as evidence; rigid-link runtime applies current mass gravity, not an identified flexible continuum',
                    'Connector mesh remains rigid visual geometry; receiving case and downstream bodies move through reduced spring',
                    'Only nominal material/geometry identified; contact and damping require physical measurements'])
        blocks.append(block);row['disposition']='Activated native reduced compliance' if placement!='sidecar_only' else 'Six-DOF matrix exported; hand mass/topology absent, not activated'
        row['block_id']=block['id']
    write(OUT/'compliance.json',dict(schema='cove50.fem-compliance-transfer.v1',blocks=blocks,inventory=inventory,
        source_review=ref(REVIEW), application='URDF is rigid reference; MJCF compliance variant adds passive coordinates without adding policy actuator channels',
        limitations='Source sidecar keeps all available connector observations. Replaced power carrier/spine and head never inherit old stiffness; reference sole/cowl/neck fields do not become invented calibrated springs.'))
    print(dict(matrices=len(blocks),activated=sum(b['placement']!='sidecar_only'for b in blocks),inventory=len(inventory)))

if __name__=='__main__': main()
