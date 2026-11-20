"""Approximate concave link geometry using existing native V-HACD runtime."""
import trimesh
import vhacdx
from common import *

def main():
    meta=read(OUT/'mesh-transfer.json')['parts']
    manifest={p['name']:p for p in read(CAD/'assembly-parts.json')['parts']}
    excluded=[p['name'] for p in meta if manifest[p['name']].get('role') in ['original_aesthetic_reference','ape_shell_reference'] and manifest[p['name']].get('include_in_physical_BOM') is False and manifest[p['name']].get('inspection_reference') is True]
    meta=[p for p in meta if manifest[p['name']].get('role')!='installation_size_scenario_not_additive_material' and p['name'] not in excluded]
    out=OUT/'collision-decomposition.json'
    links=read(out)['links'] if out.exists() else {}
    for link in list(links):
        expected={p['name'] for p in meta if p['owning_link']==link}
        if set(links[link]['source_parts'])!=expected:del links[link]
    for link in origins(read(CAD/'assembly-parts.json')):
        if link in links:continue
        meshes=[trimesh.load(OUT/p['mesh'],force='mesh')for p in meta if p['owning_link']==link]
        mesh=trimesh.util.concatenate(meshes)
        # Voxelized decomposition is collision approximation only. Full visual
        # and native moments do not use these voxelized shapes.
        if len(mesh.faces)>100000:mesh=mesh.simplify_quadric_decimation(face_count=100000)
        centre=mesh.bounds.mean(0);scale=max(mesh.extents)
        faces=np.column_stack([np.full(len(mesh.faces),3),mesh.faces]).ravel().astype(np.uint32)
        hulls=vhacdx.compute_vhacd(np.asarray((mesh.vertices-centre)/scale,np.float64),faces,
            maxConvexHulls=64 if link=='torso'else 24,resolution=400000,minimumVolumePercentErrorAllowed=.5,
            maxRecursionDepth=10,shrinkWrap=True,fillMode='surface',maxNumVerticesPerCH=64,asyncACD=True)
        rows=[]
        for i,(v,f)in enumerate(hulls):
            h=trimesh.Trimesh(v*scale+centre,f,process=False)
            filename=f'meshes/decomp_{link}_{i:02d}.stl';h.export(OUT/filename)
            rows.append(dict(mesh=filename,sha256=sha(OUT/filename),bounds_link_m=h.bounds.tolist(),volume_m3=float(h.volume)))
        links[link]=dict(hulls=rows,source_parts=[p['name']for p in meta if p['owning_link']==link],
            source_mesh_bounds_link_m=mesh.bounds.tolist(),parameters=dict(resolution=400000,max_hulls=64 if link=='torso'else 24,fill_mode='surface'))
        write(out,dict(links=links,method='V-HACD surface-voxel approximate convex decomposition of selected link meshes, SI coordinates. No native concave collision or clearance claim.',
            limits='Surface voxelization/reduction can lose thin walls, fill gaps and move contact. Source meshes and native part interfaces remain authoritative; compare native penetration/contact by configuration.'))
        print(link,len(rows),flush=True)
    # Surface-only voxelization occasionally emits a planar sliver. Native
    # engines require a volume: retain its footprint as a0.1mm contact prism.
    for link,record in links.items():
        for row in record['hulls']:
            path=OUT/row['mesh'];h=trimesh.load(path,force='mesh')
            centred=h.vertices-h.vertices.mean(0)
            _,sv,axes=np.linalg.svd(centred,full_matrices=False)
            thickness=sv[-1]/np.sqrt(len(h.vertices))
            if thickness<1e-6:
                normal=axes[-1]
                vertices=np.r_[h.vertices+normal*.00005,h.vertices-normal*.00005]
                h=trimesh.convex.convex_hull(vertices);h.export(path)
                row.update(planar_sliver_regularization_m=.0001,original_normal_rms_thickness_m=float(thickness),
                    sha256=sha(path),bounds_link_m=h.bounds.tolist(),volume_m3=float(h.volume))
    write(out,dict(links=links,excluded_unfitted_aesthetic_references=excluded,collision_semantics='Only the source-declared non-BOM unfitted aesthetic references lose hard collision; actual parts, head/neck shell and all existing physical-part exclusions remain. Conditional reference mass and visuals unchanged.',method='V-HACD surface-voxel approximate convex decomposition of selected physical link meshes; planar numerical slivers regularized to0.1mm. SI coordinates.',
        limits='Voxelization/reduction can lose thin walls, fill gaps and move contact. Per-hull regularizations disclosed. Source meshes/native part interfaces remain authoritative; no native clearance claim.'))

if __name__=='__main__':main()
