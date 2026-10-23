"""Reopen delivered CAD and STL files and record geometric export evidence."""
import json
from pathlib import Path
from build123d import import_step
import trimesh

root=Path(__file__).resolve().parent
output_root=root.parents[1]/"artifacts"/"sketch-ape-shell"
report=[]
for height in (350,500):
    folder=output_root/'exports'/f'{height}mm'
    assembly=import_step(folder/'ape_shell_assembly.step')
    solids=assembly.solids()
    bounds=assembly.bounding_box()
    source=json.loads((folder/'manifest.json').read_text())
    expected={p['name']:p['volume_mm3'] for p in source['parts']}
    volume_error=max(abs(c.volume-expected[c.label])/expected[c.label] for c in assembly.children)
    meshes={p.stem:trimesh.load_mesh(p,process=True) for p in (folder/'stl').glob('*.stl')}
    result={
        'height_requested_mm':height,
        'height_reimported_mm':bounds.size.Z,
        'dimensions_mm':list(bounds.size),
        'step_solid_count':len(solids),
        'all_step_solids_valid':all(s.is_valid and s.volume>0 for s in solids),
        'max_relative_volume_error':volume_error,
        'stl_part_count':len(meshes),
        'non_watertight_stls':[name for name,mesh in meshes.items() if not mesh.is_watertight],
    }
    assert abs(bounds.size.Z-height)<.001,result
    assert len(solids)==67 and result['all_step_solids_valid'],result
    assert volume_error<.002,result
    assert len(meshes)==67 and not result['non_watertight_stls'],result
    report.append(result)
    print(json.dumps(result),flush=True)
(root/'verification.json').write_text(json.dumps(report,indent=2))
