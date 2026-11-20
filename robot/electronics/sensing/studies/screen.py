"""Articulated optical geometry screen; no camera images, physics or installed sensing."""
from pathlib import Path
import collections
import copy
import hashlib
import json
import math
import numpy as np
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts/cove50-sensor-integration/v1'
CONFIG = ROOT / 'robot/sensor-integration/baseline-v1.json'


def read(p): return json.loads(p.read_text())
def rel(p): return str(p.relative_to(ROOT))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(p, d): p.write_text(json.dumps(d, indent=2) + '\n')


def transforms(joints, commands):
    result = {'torso': np.eye(4)}
    pending = list(joints)
    while pending:
        ready = [j for j in pending if j['parent'] in result]
        if not ready: raise ValueError('Unresolved kinematics')
        for j in ready:
            a = math.radians(commands.get(j['name'], 0))
            t = trimesh.transformations.rotation_matrix(a, j['axis_world_neutral'], j['origin_world_mm'])
            result[j['child']] = result[j['parent']] @ t
            pending.remove(j)
    return result


def compose(c):
    scene, rows, joints, controls, sources = trimesh.Scene(), [], [], [], []
    for half in ['upper', 'lower']:
        folder = ROOT / c[half]
        meta_path = folder / 'parts.json'
        model_path = folder / ('assembly-with-reference.glb' if half == 'upper' else 'assembly-with-full-shell.glb')
        m, raw = read(meta_path), trimesh.load(model_path, force='scene', process=False)
        sources.extend([dict(path=rel(p), sha256=sha(p)) for p in [meta_path, model_path]])
        index = {p['name']: p for p in m['parts']}
        for name in raw.graph.nodes_geometry:
            source_row = index.get(name)
            if source_row is None:
                source_row = next((p for p in m['parts'] if name in p.get('source_geometry_nodes', [])), None)
            if source_row is None: raise ValueError(f'Unmapped geometry {half}: {name}')
            if half == 'lower' and source_row['name'] == 'full_shell__pelvis': continue
            p = copy.deepcopy(source_row)
            t, key = raw.graph[name]
            mesh = raw.geometry[key].copy()
            mesh.apply_transform(t)
            new_name = half + '__' + name
            scene.add_geometry(mesh, node_name=new_name, geom_name=new_name)
            p.update(name=new_name, source_name=name, source_manifest=rel(meta_path), source_half=half)
            if p['owning_link'] == 'pelvis': p['owning_link'] = 'torso'
            rows.append(p)
        for j in copy.deepcopy(m['joints']):
            if j['parent'] == 'pelvis': j['parent'] = 'torso'
            joints.append(j)
        controls.extend(copy.deepcopy(m.get('controls', [])))
        # Lower pelvis reference subgroup also contains OLD fixed upper shell.
        # Recover only pelvis panels from its pinned ungrouped source.
        if half == 'lower':
            p = next(p for p in m['parts'] if p['name'] == 'full_shell__pelvis')
            path = ROOT / p['source_model']
            if sha(path) != p['source_model_sha256']: raise ValueError('Pelvis source drift')
            sources.append(dict(path=rel(path), sha256=sha(path)))
            original = trimesh.load(path, force='scene', process=False)
            for name in p['source_geometry_nodes']:
                if name not in ['ape_body_shell__pelvis_front', 'ape_body_shell__pelvis_back']: continue
                t, key = original.graph[name]
                mesh = original.geometry[key].copy(); mesh.apply_transform(t)
                new_name = 'lower__' + name
                scene.add_geometry(mesh, node_name=new_name, geom_name=new_name)
                rows.append(dict(name=new_name, source_name=name, owning_link='torso', role='shell_reference',
                                 inspection_reference=True, display_group='ape_body_shell', source_half=half,
                                 source_manifest=rel(meta_path), include_in_physical_BOM=False))
    if len({p['name'] for p in rows}) != len(rows): raise ValueError('Duplicate names')
    transforms(joints, {})
    return scene, dict(parts=rows, joints=joints, controls=controls, root_link='torso',
                      units='GLB m; origins mm; angles deg', scope='Immutable upper V5 / lower V16 sensing context; fixed common root, no new center structure, dynamics or global selection.'), sources


def target_points(center, width, height):
    x,y,z = center
    return np.array([center, *[[x+dx*width/2,y,z+dz*height/2] for dx in [-1,1] for dz in [-1,1]]], float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = read(CONFIG)
    selection = read(ROOT/'robot/design.json')['section_selections']
    for record in [selection['Torso / shell'], selection['Ankle / foot']['connected_assembly_actuator_override']]:
        for item in record['source'].values():
            if 'sha256' in item and sha(ROOT/item['path']) != item['sha256']: raise ValueError('Selected source drift')
    scene, manifest, sources = compose(c)
    scene.export(OUT/'assembly.glb'); write(OUT/'parts.json', manifest)
    print(f'Composed {len(manifest["parts"])} groups / {len(manifest["joints"])} joints', flush=True)
    # Build acceleration once per rigid link, in neutral mm. Face IDs preserve blocker identity.
    by_link = collections.defaultdict(list)
    for p in manifest['parts']:
        if p['source_name'] == c['camera']['source_part']: continue  # lens starts outside housing
        t,key = scene.graph[p['name']]; mesh = scene.geometry[key].copy(); mesh.apply_transform(t); mesh.apply_scale(1000)
        by_link[p['owning_link']].append((p,mesh))
    intersectors = {}
    for link, entries in by_link.items():
        merged = trimesh.util.concatenate([m for _,m in entries])
        ids = np.concatenate([np.full(len(m.faces),i) for i,(_,m) in enumerate(entries)])
        intersectors[link] = (RayMeshIntersector(merged),ids,[p for p,_ in entries])
    targets = []
    for distance in [120,250,400,800]:
        for x in [-150,0,150]:
            targets.append(dict(name=f'floor_{distance}_x{x}',center_mm=[x,-distance,40],width_mm=80,height_mm=80))
    for name, center, size in [('shown_object',[0,-350,445],70),('seated_face',[0,-1000,1000],160),('adult_face',[0,-1000,1600],160)]:
        targets.append(dict(name=name,center_mm=center,width_mm=size,height_mm=size))
    # Target rigidly accompanies each actual forearm; a provisional 30mm object beyond its frontmost neutral surface.
    for side in ['R','L']:
        meshes=[m for p,m in by_link[side+'_forearm']]
        b=trimesh.util.concatenate(meshes).bounds
        targets.append(dict(name=side+'_near_hand',center_mm=[float(b.mean(0)[0]),float(b[0,1]-30),float(b.mean(0)[2])],width_mm=30,height_mm=30,link=side+'_forearm',basis='30mm beyond actual forearm front bound, not final wrist/grasp target'))
    arm_poses=[('arms_neutral',{}),('bilateral_forward', {s+'_shoulder_pitch':90 for s in ['L','R']}),('right_bent_reach',{'R_shoulder_pitch':90,'R_elbow':90})]
    poses=[dict(name=f'{family}_yaw{yaw}_pitch{pitch}',commands_deg={**arms,'neck_yaw':yaw,'neck_pitch':pitch},family=family,yaw=yaw,pitch=pitch)
           for family,arms in arm_poses for yaw in [-60,-30,0,30,60] for pitch in [-30,-15,0,15,35]]
    write(OUT/'config.json',dict(sensor_config=c,targets=targets,poses=poses,
          assumptions=['Fixed torso and neutral lower joints. No walking/balance or lower sweep is simulated.', 'Opaque triangle surfaces including design-reference shells; no manufactured final shell inferred.', 'Source lens-ring centers and nominal full-sensor fields; actual crop, distortion and entrance pupil uncalibrated.', 'Camera housing omitted from occluders; window/cover not modeled.'],
          source_hashes=sources+[dict(path=rel(CONFIG),sha256=sha(CONFIG)),dict(path=rel(Path(__file__)),sha256=sha(Path(__file__)))]))
    summaries=[]
    with (OUT/'ray-trace.jsonl').open('w') as trace:
        for pose in poses:
            tf=transforms(manifest['joints'],pose['commands_deg'])
            for view in c['camera']['views']:
                head=tf['head'];origin=trimesh.transform_points([view['origin_mm']],head)[0]
                basis=head[:3,:3] @ np.array(c['camera']['optical_to_head_rotation'])
                points=np.concatenate([trimesh.transform_points(target_points(t['center_mm'],t['width_mm'],t['height_mm']),tf[t.get('link','torso')]) for t in targets])
                vec=points-origin; distance=np.linalg.norm(vec,axis=1);dirs=vec/distance[:,None]
                camera=vec @ basis
                angle_h=np.degrees(np.arctan2(camera[:,0],camera[:,2]));angle_v=np.degrees(np.arctan2(camera[:,1],camera[:,2]))
                inside=(camera[:,2]>0)&(np.abs(angle_h)<=view['hfov_deg']/2)&(np.abs(angle_v)<=view['vfov_deg']/2)
                nearest=np.full(len(points),np.inf);blockers=[None]*len(points);blocker_reference=[None]*len(points)
                for link,(inter,ids,entries) in intersectors.items():
                    inv=np.linalg.inv(tf[link]);o=trimesh.transform_points([origin],inv)[0];d=dirs @ inv[:3,:3].T
                    loc,ri,fi=inter.intersects_location(np.repeat(o[None,:],len(points),axis=0),d,multiple_hits=True)
                    if not len(ri): continue
                    ds=np.linalg.norm(loc-o,axis=1)
                    for ray,face,dist in zip(ri,fi,ds):
                        if 0.05<dist<distance[ray]-0.1 and dist<nearest[ray]:
                            nearest[ray]=float(dist);part=entries[ids[face]];blockers[ray]=part['name'];blocker_reference[ray]=bool(part.get('inspection_reference'))
                for ti,target in enumerate(targets):
                    sl=slice(ti*5,(ti+1)*5);visible=inside[sl]&~np.isfinite(nearest[sl])
                    summary=dict(pose=pose['name'],family=pose['family'],yaw=pose['yaw'],pitch=pose['pitch'],view=view['name'],target=target['name'],in_frame_points=int(sum(inside[sl])),unoccluded_in_frame_points=int(sum(visible)),sample_points=5)
                    summaries.append(summary)
                    for k in range(5):
                        i=ti*5+k
                        trace.write(json.dumps(dict(**summary,sample_index=k,origin_world_mm=origin.tolist(),target_world_mm=points[i].tolist(),distance_mm=float(distance[i]),horizontal_angle_deg=float(angle_h[i]),vertical_angle_deg=float(angle_v[i]),in_frame=bool(inside[i]),blocker=blockers[i],blocker_is_reference=blocker_reference[i],first_hit_distance_mm=float(nearest[i]) if math.isfinite(nearest[i]) else None,visible=bool(visible[k])))+'\n')
            if pose['yaw']==60: print('Screened '+pose['name'],flush=True)
    write(OUT/'view-summary.json',summaries)
    report=dict(configurations=len(poses),rays=len(summaries)*5,targets=len(targets),views=3,full_trace=rel(OUT/'ray-trace.jsonl'),source_hashes=sources,
                registration='Identity common frame. Lower pelvis reference filtered to actual two pelvis panels, eliminating obsolete fixed upper shell. No new center-body connection designed.',
                groups=len(manifest['parts']),joints=len(manifest['joints']),
                grades={'source_registration':dict(grade=3.5,confidence='high',trend='new baseline'),'optical_coverage':dict(grade=2,confidence='low',trend='new uncalibrated geometric screen'),'installed_timing':dict(grade=0.5,confidence='high',trend='contract only'),'contact_sensing':dict(grade=0,confidence='high',trend='unselected hardware'),'harness':dict(grade=0.5,confidence='high',trend='requirements only')},
                limitations=['Ray visibility is neither image recognition nor stereo depth validity. Five points do not measure full object visibility.', '75 sampled kinematic configurations, not continuous motion, calibrated optics or measured latency.', 'V5/V16 center-body attachment, final shells, USB harness, fixed IMU and contact packages remain unresolved.','Head/stereo features are CAD-derived exterior origins, not measured entrance pupils; actual intrinsics/extrinsics/crop and focus revision unknown.'])
    write(OUT/'report.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
