"""Publish current physics/perception evidence and its source CAD characterization."""
import shutil
from common import *

DESIGN='cove50-selected-learning-body-r1-20261026'

def main():
    report=read(OUT/'report.json');replay=read(OUT/'replay-summary.json')
    for p in (ROOT/'robot/model').rglob('*'):
        if p.is_file() and '__pycache__'not in p.parts:
            q=OUT/'authored-source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for p in DATA.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.md'] and not p.name.startswith(('preservation-','source-commit','transport-')):
            q=OUT/'coordination'/p.relative_to(DATA);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    paths={p for p in OUT.rglob('*')if p.is_file() and p.name not in ['publication.json','design-publication.json']}
    # Exact selected CAD stays the sole displayed scene, without another selection.
    paths.update(CAD/f for f in ['inspection.glb','assembly-parts.json','assembly.glb','source-lock.json'])
    def dependencies(value):
        if isinstance(value,dict):
            path=value.get('path')
            if isinstance(path,str) and path.startswith('artifacts/') and (ROOT/path).is_file():paths.add(ROOT/path)
            for child in value.values():dependencies(child)
        elif isinstance(value,list):
            for child in value:dependencies(child)
    for name in ['source-lock.json','part-mass-ledger.json','compliance.json','FEM-geometry-identity.json','actuators.json','sensors.json']:
        dependencies(read(OUT/name))
    assets=[]
    for p in sorted(paths):
        name=rel(p);assets.append(dict(id='model-'+hashlib.sha256(name.encode()).hexdigest()[:20],title=p.name,path=name,
            kind='model'if p.suffix=='.glb'else'image'if p.suffix.lower()in['.png','.jpg','.jpeg','.webp']else'data'))
    refs=[dict(label=label,path=rel(OUT/name))for label,name in [
        ('Model report and uncertainty','report.json'),('Download self-contained robot description','cove50-description-r1-20261026.zip'),
        ('Exact geometry and source identities','source-lock.json'),('Link mass and full inertia','link-inertias.json'),
        ('Supplier actuator data and assumptions','actuators.json'),('FEM mappings and full coupled stiffness','compliance.json'),
        ('Sensor extrinsics and nominal intrinsics','sensors.json'),('Capture, transport and compute timing','timing.json'),
        ('Actual native model validation','native-validation.json'),('Numerical sensitivity','numerical-sensitivity.json'),
        ('Portable consumer verification','portable-consumer-verification.json')]]
    clips=[dict(id='cove50-model-r1-final-'+r['id'],title=r['id'],role='Complete native episode; loss of upright balance',
        body='Selected V24/R23/R33 CAD; '+r['variant'],observation='Assumed closed-loop servos follow a retargeted old attained-position trace; no new actor inference.',
        source_path=rel(OUT/'replay'/f"{r['id']}.mp4"),report_path=r['trace']['path'].replace('.npz','.json'),trace_path=r['trace']['path'])for r in replay['episodes']]
    limits='Provisional model: estimated mass/internal inertia, unknown hip stops/zeros, unmeasured servo/contact/thermal/latency/calibration. Flexible numerical/contact reduction remains sensitive. No transferred walking policy, physical localization or qualification.'
    record=dict(id=RESULT,title='Selected Cove-50 learning and perception model',dashboard='learning',workspace='Control',
        area='CAD-to-learning body transfer and perception frames',status='provisional_model',observation=report['observation'],
        measurements=f"22 actuators; 13 sensor frames; 16 active coupled FEM interfaces; estimated {report['mass']['total_mass_kg']:.4f} kg; 6 primary episodes plus 3 numerical probes; 0 new policies.",
        limitations=limits,body='Selected electronics V24/R23/R33; upper V5/lower V16',provenance=[rel(OUT/'source-lock.json')],
        matrix=report['matrix'],grades=report['grades'],assets=assets,clips=clips,
        sections=[dict(title='Perception and capture-time localization contract',payload=dict(kind='analysis',
            observation='61 prescribed head captures with native RGB/depth; capture-time transforms and unavailable packet conditions retained.',
            latency_conditions=report['perception']['latency_conditions'],source_refs=refs[6:9],limitations='Synthetic truth supplied for transform verification; physical calibration and estimator unqualified.'))])
    write(OUT/'publication.json',record)
    engineering=dict(record,id=DESIGN,title='Selected body · learning-model characterization',dashboard='design',workspace='Body',
        study_id='cove50-selected-learning-body',version=VERSION,robot_sections=['Whole robot','Electronics'],clips=[],
        area='Mass, kinematics and FEM model characterization',status='provisional_characterization',sections=[])
    engineering['design']=dict(id=DESIGN,title='Selected body characterization',summary='Physics and sensor description of the existing selected integrated electronics geometry.',
        visual_kind='native_CAD_assembly',maturity='Provisional learning model; physical calibration open',selected=False,
        image_path=rel(OUT/'model-neutral.png'),description=report['observation'],
        cad=dict(model_units='m',motion_mode='saved_configurations',views=[dict(label='Existing selected CAD · source neutral pose',model_path=rel(CAD/'inspection.glb'))],
            parts_path=rel(CAD/'assembly-parts.json'),report_path=rel(OUT/'report.json'),has_body_reference=True,
            scope='Source CAD frame shown here; physics uses proper (-CADY,CADX,CADZ) rotation. Source Electronics study remains selected. No new CAD or whole-body mechanical selection.'),
        source_refs=refs,evidence=[dict(id='physics-characterization-r1',kind='simulation',title='Source geometry, mass and elastic model',
            observation='853scene entries retained, estimated mass with full native inertia,22source axes,18exported/16active FEM matrices. Rigid reference and separate flexible MJCF.',
            measurements=[dict(name='Conditional robot mass',value=report['mass']['total_mass_kg'],unit='kg'),dict(name='Unrepresented installation allowance',value=150,unit='g'),
                dict(name='Actuated axes',value=22,unit='axes'),dict(name='Coupled active FEM interfaces',value=16,unit='interfaces')],
            conditions=dict(source='Selected V24/R23/R33; upperV5/lowerV16',assumptions='Explicit source-by-source mass, physical and timing uncertainty'),
            limitations=limits,source_refs=refs)])
    write(OUT/'design-publication.json',engineering)
    print(dict(learning=RESULT,design=DESIGN,assets=len(assets),clips=len(clips)))

if __name__=='__main__':main()
