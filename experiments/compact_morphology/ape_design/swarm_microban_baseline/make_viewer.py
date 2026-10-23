"""Generate a shell and simple-envelope GLB viewer from the comparison report."""
import json
from pathlib import Path
import numpy as np
import build123d as b
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts/microban-baseline'
OUT.mkdir(parents=True,exist_ok=True)
# Rebuild the exact 500 mm source shell and retain each skin piece as a GLTF node.
import sys
sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
from build import Study
sys.path.insert(0,str(Path(__file__).resolve().parent))
from microban_baseline import plane_shape
st=Study(500,2,.65);st.build()
skins=[p['shape'] for p in st.parts if p['name']!='chest_indicator']
for s in skins:s.color=b.Color(0.72,0.76,0.79,0.26)
b.export_gltf(b.Compound(children=skins),OUT/'shell.glb',binary=True,linear_deflection=.5,angular_deflection=.25)
report=json.load(open(OUT/'report.json'))
for variant,v in report['variants'].items():
    boxes=[]
    for row in v['component_rows']:
        g=row['geometry']; dims=g['size_mm']; center=g.get('center_mm')
        if center is None: continue
        R=g.get('rotation',[[1,0,0],[0,1,0],[0,0,1]])
        box=plane_shape(center,np.asarray(R),dims)
        boxes.append(box)
        box.color=b.Color(0.9,0.25,0.12,0.8) if row['shell_intersections'] else b.Color(0.15,0.35,0.9,0.8)
    b.export_gltf(b.Compound(children=boxes),OUT/f'{variant}.glb',binary=True,linear_deflection=.5,angular_deflection=.25)
print('Wrote shell.glb and 3 diagnostic variants')
