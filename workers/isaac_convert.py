"""Robot-specific options around Isaac Lab's maintained MJCF converter.

Run inside the pinned Linux/NVIDIA Isaac Lab environment. Conversion is not
loaded-articulation, contact, controller or physical qualification.
"""
import argparse
import json
from pathlib import Path
import traceback


def conversion_options(model, output):
    if model.name not in {'robot.xml', 'supported.xml'}:
        raise ValueError('Expected a generated free or supported robot model')
    return dict(asset_path=str(model.resolve()),usd_dir=str(output.resolve()),
                usd_file_name=model.stem+'.usd',force_usd_conversion=True,
                merge_mesh=False,collision_from_visuals=False,self_collision=True,
                fix_base=model.name=='supported.xml',link_density=0.0,
                physics_variant='physics',run_asset_transformer=True,
                run_multi_physics_conversion=True,debug_mode=True)


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.physics import PhysicsCfg
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from isaaclab.utils.version import standalone_importers_available
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model', choices=['robot.xml','supported.xml'], action='append', help='Convert only the requested model(s); default preserves the two-model pilot')
    add_launcher_args(parser);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='fail',scope=__doc__,models=[],articulation_parity='not_run',controller_parity='not_run')
    try:
        if not standalone_importers_available():raise RuntimeError('Install the pinned importers extra first')
        args.require_kit=False;args.physics='newton_mjwarp'
        with launch_simulation(cfg=PhysicsCfg(),launcher_args=args):
            for name in args.model or ['robot.xml','supported.xml']:
                cfg=conversion_options(args.bundle/'model'/name,args.output/Path(name).stem)
                converted=MjcfConverter(MjcfConverterCfg(**cfg))
                path=Path(converted.usd_path).resolve()
                if not path.is_file() or not path.is_relative_to(args.output.resolve()):raise ValueError('Missing or external converter output')
                report['models'].append(dict(input=name,options=cfg,usd_path=str(path.relative_to(args.output.resolve()))))
        report['status']='pass'
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
