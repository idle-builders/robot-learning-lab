"""Compare retained world predictions to their observed clip without inventing rewards."""
import argparse
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
from lab.artifact_store import plan_migration,apply_migration,verify_manifest,snapshot_file


def review(run):
    run=Path(run);prediction=run/'evidence/prediction'
    report=json.loads((prediction/'report.json').read_text())
    output=ROOT/'artifacts'/('world-model-review-'+uuid.uuid4().hex[:12]);output.mkdir()
    def read(path):
        reader=imageio.get_reader(path);metadata=reader.get_meta_data();frames=list(reader.iter_data());reader.close()
        return frames,metadata
    observed,meta=read(prediction/'inputs/reference.mp4')
    clips={};metrics={}
    for case in report['cases']:
        mode=case['mode'];frames,metadata=read(prediction/case['videos'][0]);clips[mode]=frames
        if len(frames)!=13 or metadata['fps']!=3:raise ValueError('Unexpected prediction timeline')
        errors=[];psnr=[]
        for predicted,actual in zip(frames,observed[:13],strict=True):
            actual=np.asarray(Image.fromarray(actual).resize((predicted.shape[1],predicted.shape[0]),Image.Resampling.BILINEAR),dtype=float)/255
            diff=actual-np.asarray(predicted,dtype=float)/255
            errors.append(float(np.mean(abs(diff))));psnr.append(float(-10*np.log10(max(float(np.mean(diff**2)),1e-12))))
        metrics[mode]={'frame_count':len(frames),'width':metadata['size'][0],'height':metadata['size'][1],
            'fps':metadata['fps'],'per_frame_rgb_mae_0_to_1':errors,'per_frame_psnr_db':psnr,
            'comparison_scope':'recorded-action visual prediction error' if mode=='recorded' else 'difference from original recording; NOT counterfactual accuracy'}
    tile_w,tile_h=320,256
    sheet=Image.new('RGB',(tile_w*3,(tile_h+25)*3),'white');draw=ImageDraw.Draw(sheet)
    for row,(mode,frames) in enumerate([('observed',observed),*clips.items()]):
        for col,index in enumerate([0,6,12]):
            pic=Image.fromarray(frames[index]);pic.thumbnail((tile_w,tile_h))
            sheet.paste(pic,(col*tile_w,row*(tile_h+25)+25))
            draw.text((col*tile_w+5,row*(tile_h+25)+5),f'{mode}: frame {index}',fill='black')
    sheet.save(output/'comparison.jpg')
    detail={'status':'requires_visual_review','source_run':str(run),'input_source':report.get('input_source','bridge'),
        'model_revision':report['model_revision'],'checkpoint_sha256':report['model_file_sha256'],
        'metrics':metrics,'world_model_environment':'not_qualified','imagined_rl':'not_run',
        'notes':'Pixel error is an inspection diagnostic, not a calibrated physical reward or proof of contact accuracy. Changed actions have no observed counterfactual.'}
    (output/'report.json').write_text(json.dumps(detail,indent=2))
    snapshot_file(Path(__file__),output/'review-source.py')
    migration=plan_migration(ROOT,[output]);apply_migration(ROOT,migration);verify_manifest(ROOT,migration)
    print(output)
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args();review(args.run)
