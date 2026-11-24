"""Seal a self-contained description package, retaining source provenance."""
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET
from common import *

def main():
    source=Path(__file__).parent
    runtime_dir=OUT/'runtime';runtime_dir.mkdir(exist_ok=True)
    for name in ['common.py','runtime.py','mjlab_adapter.py']:
        shutil.copyfile(source/name,runtime_dir/name)
    shutil.copyfile(source.parent/'README.md',OUT/'README.md')
    (OUT/'package.xml').write_text('''<?xml version="1.0"?>
<package format="3">
  <name>cove50_description</name>
  <version>0.1.0</version>
  <description>Selected APE0 CAD learning description; provisional physical parameters.</description>
  <maintainer email="maintainer@example.invalid">Robot Learning Lab maintainer</maintainer>
  <license>SEE SOURCE PROVENANCE</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <export><build_type>ament_cmake</build_type></export>
</package>
''')
    (OUT/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.8)
project(cove50_description)
find_package(ament_cmake REQUIRED)
install(DIRECTORY meshes DESTINATION share/${PROJECT_NAME})
install(FILES cove50.ros.urdf cove50.srdf DESTINATION share/${PROJECT_NAME}/urdf)
ament_package()
''')
    payload=[OUT/name for name in ['cove50.urdf','cove50.ros.urdf','cove50.srdf','cove50-rigid.xml','cove50-compliant.xml',
        'actuators.json','sensors.json','timing.json','frames.json','compliance.json','compliance-mass-splits.json','part-mass-ledger.json',
        'link-inertias.json','mesh-transfer.json','collision-model.json','collision-decomposition.json','model-build.json','source-lock.json',
        'README.md','package.xml','CMakeLists.txt','power-and-randomization.json','report.json','current-model.json','mjlab-entity-interface.json']]
    payload+=list(runtime_dir.glob('*.py'))
    meshes=set()
    for name in ['cove50.urdf','cove50-rigid.xml','cove50-compliant.xml']:
        for node in ET.parse(OUT/name).iter('mesh'):
            path=node.get('filename',node.get('file'))
            if path:meshes.add(OUT/path)
    payload+=sorted(meshes)
    inventory=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size)for p in sorted(set(payload))]
    write(OUT/'package-inventory.json',dict(schema='cove50.description-package.v1',version=VERSION,files=inventory,
        source_provenance='Original supplier CAD/documents and FEM fields remain separately source-hashed in private evidence. No new license to redistribute third-party source geometry is inferred.',
        build_recipe=ref(Path(__file__))))
    payload.append(OUT/'package-inventory.json')
    archive=OUT/f'cove50-description-{VERSION}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(set(payload)):
            entry=zipfile.ZipInfo('cove50_description/'+str(p.relative_to(OUT)),date_time=(2026,10,26,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED;entry.external_attr=0o100644<<16
            z.writestr(entry,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=4)
    with zipfile.ZipFile(archive) as z:
        mismatches=[]
        for row in inventory:
            data=z.read('cove50_description/'+row['path'])
            if hashlib.sha256(data).hexdigest()!=row['sha256']:mismatches.append(row['path'])
    write(OUT/'package-verification.json',dict(archive=ref(archive),files=len(payload),mesh_dependencies=len(meshes),
        uncompressed_bytes=sum(p.stat().st_size for p in payload),hash_mismatches=mismatches))
    print(dict(files=len(payload),meshes=len(meshes),zip_bytes=archive.stat().st_size))

if __name__=='__main__':main()
