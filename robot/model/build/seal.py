"""Freeze authored recipes, environments and native dependency identities."""
import importlib.metadata,platform,shutil,sys,xml.etree.ElementTree as ET
from common import *

def main():
    for p in (ROOT/'robot/model').rglob('*'):
        if p.is_file()and'__pycache__'not in p.parts:
            q=OUT/'authored-source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    execution=OUT/'execution';execution.mkdir(exist_ok=True)
    for name in ['contract','build','replay','validation','perception','mjlab','numerical','render']:
        p=Path('/tmp')/f'model-{name}-final.log'
        if p.exists():shutil.copyfile(p,execution/p.name)
    versions={}
    for name in ['mujoco','numpy','scipy','trimesh','build123d','vhacdx','fast-simplification']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    models=[];meshpaths=set()
    for name in ['cove50.urdf','cove50.ros.urdf','cove50-rigid.xml','cove50-compliant.xml']:
        path=OUT/name;models.append(ref(path))
        for n in ET.parse(path).iter('mesh'):
            filename=n.get('file',n.get('filename','')).replace('package://cove50_description/','')
            if filename:meshpaths.add(OUT/filename)
    sources=[ref(p)for p in sorted((ROOT/'robot/model').rglob('*'))if p.is_file()and'__pycache__'not in p.parts]
    record=dict(schema='cove50.learning-model.seal.v1',version=VERSION,environment=dict(python=sys.version,platform=platform.platform(),packages=versions),
        authored_source=sources,models=models,meshes=[ref(p)for p in sorted(meshpaths)],
        contracts=[ref(OUT/name)for name in ['source-lock.json','frames.json','actuators.json','sensors.json','timing.json','part-mass-ledger.json','link-inertias.json','compliance.json','collision-decomposition.json','report.json']],
        full_native_reports=[ref(OUT/name)for name in ['native-validation.json','replay-summary.json','numerical-sensitivity.json','perception-validation.json','mjlab-entity-interface.json','portable-consumer-verification.json']],
        qualification='Exact file identities and consuming-path observations, not measured physical fidelity or policy qualification.')
    write(OUT/'evidence-seal.json',record)
    write(DATA/'verification.json',dict(models=record['models'],mesh_dependencies=len(meshpaths),source_files=len(sources),
        seal=ref(OUT/'evidence-seal.json'),portable_package=ref(OUT/f'cove50-description-{VERSION}.zip'),
        checks=['Native URDF/rigid/flexible compilation and FK comparison','Independent native six-DOF spring coupons',
            'Six complete primary episodes and three independent numerical probes; full traces and falls retained',
            'Native RGB/depth capture, projection/timing and invalid packet conditions','Actual MJLab Entity compile',
            'Extracted package Runtime/native/MJLab consumers; no external meshes'],grades=read(OUT/'report.json')['grades']))
    print(dict(sources=len(sources),mesh_dependencies=len(meshpaths)))

if __name__=='__main__':main()
