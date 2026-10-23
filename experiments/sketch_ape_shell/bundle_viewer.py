"""Bundle the actual 350 mm CAD mesh and Three.js into an offline HTML viewer."""
import base64
from pathlib import Path

root=Path(__file__).resolve().parent
output_root=root.parents[1]/"artifacts"/"sketch-ape-shell"

def data_module(text):
    return 'data:text/javascript;base64,'+base64.b64encode(text.encode()).decode()

import json
imports={
    'three-core':data_module((root/'vendor/three.core.js').read_text()),
    'three':data_module((root/'vendor/three.module.js').read_text().replace('./three.core.js','three-core')),
    'orbit-controls':data_module((root/'vendor/OrbitControls.js').read_text()),
}
html=(root/'viewer.html').read_text()
html=html.replace('{"imports":{"three":"./vendor/three.module.js"}}',json.dumps({'imports':imports}))
html=html.replace("from './vendor/OrbitControls.js'","from 'orbit-controls'")
html=html.replace("const data=await (await fetch('./exports/350mm/mesh.json')).json();",
                  "const data=JSON.parse(document.getElementById('cad-data').textContent);")
mesh=(output_root/'exports/350mm/mesh.json').read_text()
html=html.replace('<script type="module">','<script id="cad-data" type="application/json">'+mesh+'</script>\n<script type="module">')
output_root.mkdir(parents=True,exist_ok=True)
(output_root/'preview.html').write_text(html)
print(output_root/'preview.html')
