"""Orchestrate native conversion, reference, upstream PPO and checkpoint reload."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=50)
a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
root=Path(__file__).parent;out=a.output.resolve();bundle=a.bundle.resolve()
os.environ.update(ROBOTLAB_BUNDLE=str(bundle),ROBOTLAB_SETTINGS=str(root/'settings.json'),PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
report={'status':'partial','stages':[],'vision':'not_run','higher_level_rl':'not_run','gaussian_gym':'not_run'}
request=json.loads((root/'request.json').read_text())
if request.get('dynamics'):
    os.environ['ROBOTLAB_ELECTROMECHANICAL']=json.dumps(request['dynamics'])
def run(name,cmd):
    with (out/(name+'.log')).open('xb') as log:
        result=subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    report['stages'].append({'stage':name,'exit_code':result.returncode})
    if result.returncode: raise RuntimeError(name+' failed')
try:
    if request.get('dynamics'):
        run('actuator-probe',[sys.executable,str(root/'actuator_probe.py'),'--output',str(out/'actuator-probe')])
        probe=json.loads((out/'actuator-probe/report.json').read_text())
        report['actuator_parity']=probe['native'].get('parity_status','not_run')
        if report['actuator_parity']!='pass':raise RuntimeError('Native actuator/reference comparison failed')
    run('convert',[sys.executable,str(root/'isaac_convert.py'),'--bundle',str(bundle),'--output',str(out/'converted'),'--model','robot.xml','--viz','none'])
    conv=json.loads((out/'converted/report.json').read_text())
    usd=next(x for x in conv['models'] if x['input']=='robot.xml')['usd_path']
    # Keep the imported world floor out of the articulation instance. Otherwise
    # its static plane is translated by the robot spawn pose into the robot.
    from pxr import Usd, UsdGeom, UsdPhysics
    imported=Usd.Stage.Open(str(out/'converted'/usd))
    root_name=imported.GetDefaultPrim().GetName()
    isolated=Usd.Stage.CreateNew(str(out/'articulation.usda'))
    UsdGeom.SetStageMetersPerUnit(isolated,1.0)
    UsdGeom.SetStageUpAxis(isolated,UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(isolated,1.0)
    prim=isolated.DefinePrim('/'+root_name,'Xform')
    prim.GetReferences().AddReference('./converted/'+usd)
    isolated.SetDefaultPrim(prim)
    floor=isolated.GetPrimAtPath('/'+root_name+'/Geometry/floor')
    if not floor.IsValid():raise RuntimeError('Expected imported world floor was not found')
    floor.SetActive(False)
    isolated.GetRootLayer().Save()
    os.environ['ROBOTLAB_USD']=str(out/'articulation.usda')
    if request['stage'] in ('camera','skills'):
        motor_checkpoint=root/'motor.pt'
        if request.get('retrain_motor'):
            run('train',[str(Path(sys.executable).with_name('isaaclab')),'train','--task','RobotLab-PipDrive-v0','--external_callback','robotlab.register','--num_envs','64','--viz','none','--rl_library','rsl_rl','--max_iterations',str(a.iterations),'--seed','17'])
            checkpoints=sorted(out.glob('logs/**/*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
            if not checkpoints:raise RuntimeError('No saved motor checkpoint')
            motor_checkpoint=checkpoints[-1]
            report['robot_rl']='executed';report['checkpoint']=str(motor_checkpoint.relative_to(out))
        os.environ['ROBOTLAB_MOTOR_CHECKPOINT']=str(motor_checkpoint)
        if request.get('selector_checkpoint_sha256'):
            import hashlib
            selector=root/'selector.zip'
            if hashlib.sha256(selector.read_bytes()).hexdigest()!=request['selector_checkpoint_sha256']:raise ValueError('Selector checkpoint hash mismatch')
            os.environ['ROBOTLAB_SELECTOR_CHECKPOINT']=str(selector)
        if request.get('verify_dynamics'):
            run('reference',[sys.executable,str(root/'evaluate.py'),'--output',str(out/'reference.json'),'--viz','none'])
            run('evaluate',[sys.executable,str(root/'evaluate.py'),'--checkpoint',str(motor_checkpoint),'--output',str(out/'evaluation.json'),'--viz','none'])
            report['dynamics_evaluation']=json.loads((out/'evaluation.json').read_text())['status']
        run('renderer-dependencies',['uv','sync','--project',str(Path(sys.executable).parents[2]),'--python',sys.executable,'--frozen','--extra','importers','--extra','ovrtx','--extra','sb3'])
        run('camera-dependencies',[sys.executable,'-m','ensurepip'])
        run('install-camera',[sys.executable,'-m','pip','install','opencv-contrib-python-headless==4.13.0.92','--no-deps','--timeout','20','--retries','1'])
        if request.get('gaussian'):
            run('gaussian',[sys.executable,str(root/'gaussian.py'),'--output',str(out/'gaussian')])
            if request.get('capture_baseline',True):
                run('camera-baseline',[sys.executable,str(root/'camera.py'),'--output',str(out/'camera-baseline'),'--viz','none'])
                report['baseline_vision']=json.loads((out/'camera-baseline/report.json').read_text())['status']
                if report['baseline_vision']!='pass':raise RuntimeError('Baseline head-camera target detection failed')
            from scene import create_gaussian_scene
            os.environ['ROBOTLAB_GAUSSIAN_USD']=str(create_gaussian_scene(out/'gaussian'))
            report['gaussian_export']='pass'
        run('camera',[sys.executable,str(root/'camera.py'),'--output',str(out/'camera'),'--viz','none',*(['--skills'] if request['stage']=='skills' else [])])
        report['vision']=json.loads((out/'camera/report.json').read_text())['status']
        if request.get('gaussian'):
            import cv2,numpy as np
            fraction=None
            if (out/'camera-baseline/camera.png').exists():
                before=cv2.imread(str(out/'camera-baseline/camera.png'))
                after=cv2.imread(str(out/'camera/camera.png'))
                fraction=float(np.mean(np.max(np.abs(before.astype(float)-after.astype(float)),axis=2)>8))
            stage=Usd.Stage.Open(os.environ['ROBOTLAB_GAUSSIAN_USD'])
            fields=[str(p.GetPath()) for p in stage.Traverse() if 'GaussianSplat' in str(p.GetTypeName())]
            placement=json.loads((out/'camera/gaussian-placement.json').read_text())
            comparison=dict(status='needs_image_review' if fields and placement['status']=='pass' else 'fail',particle_fields=fields,changed_pixel_fraction=fraction,native_placement=placement['status'],scope='Image differences can be rendering noise; never promote visibility from this scalar alone. Inspect native camera images. No collision/reconstruction/room claim.')
            (out/'gaussian-camera-comparison.json').write_text(json.dumps(comparison,indent=2))
            report['gaussian_gym']=comparison['status']
        if request['stage']=='skills':
            report['higher_level_rl']=json.loads((out/'camera/skills/report.json').read_text())
    else:
        run('reference',[sys.executable,str(root/'evaluate.py'),'--output',str(out/'reference.json'),'--viz','none'])
        run('train',[str(Path(sys.executable).with_name('isaaclab')),'train','--task','RobotLab-PipDrive-v0','--external_callback','robotlab.register','--num_envs','64','--viz','none','--rl_library','rsl_rl','--max_iterations',str(a.iterations),'--seed','17'])
        checkpoints=sorted(out.glob('logs/**/*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        if not checkpoints:raise RuntimeError('No saved checkpoint')
        run('evaluate',[sys.executable,str(root/'evaluate.py'),'--checkpoint',str(checkpoints[-1]),'--output',str(out/'evaluation.json'),'--viz','none'])
        report['robot_rl']='executed';report['checkpoint']=str(checkpoints[-1].relative_to(out))
        report['held_out_status']=json.loads((out/'evaluation.json').read_text())['status']
except Exception as exc:
    report.update(status='fail',error=repr(exc))
finally:
    (out/'report.json').write_text(json.dumps(report,indent=2))
raise SystemExit(report['status']=='fail')
