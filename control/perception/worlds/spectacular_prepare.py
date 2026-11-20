"""Export full-quality Spectacular keyframes for the existing Gaussian trainer.

Uses the SDK Replay/Mapping API with upstream offline iOS mono configuration.
All keyframe observations are retained. RGB holdouts condition on the joint
sensor camera solve; they are not capture-disjoint localization evaluation.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import spectacularAI


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--keyframe-distance',type=float)
    p.add_argument('--fast',action='store_true')
    p.add_argument('--width',type=int,default=720)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    (out/'dataset/images').mkdir(parents=True,exist_ok=True)
    calibration=json.loads((a.input/'calibration.json').read_text())
    if any(c['model']!='pinhole' for c in calibration['cameras']):
        raise ValueError('This adapter retains native pinhole images; another camera model needs explicit rectification')
    cache=out.parent.parent/'sdk-image-cache';cache.mkdir(exist_ok=True)
    rows=[json.loads(l) for l in (a.input/'data.jsonl').read_text().splitlines()]
    source_frames=[r for r in rows if 'frames' in r]
    frame_times=np.array([r['time'] for r in source_frames]);start=frame_times[0]
    config={'maxMapSize':0,'useSlam':True,'useStereo':False,'passthroughColorImages':True,
            'keyframeDecisionDistanceThreshold':.05,'icpVoxelSize':.05,
            'maxKeypoints':1000,'optimizerMaxIterations':30,
            'parameterSets':['wrapper-base','offline-base','ios-tof']}
    if a.fast:
        config['parameterSets']=['wrapper-base','ios-tof'];config['keyframeDecisionDistanceThreshold']=.1
        config['icpVoxelSize']=.1;config.pop('maxKeypoints');config.pop('optimizerMaxIterations')
    if a.keyframe_distance is not None:config['keyframeDecisionDistanceThreshold']=a.keyframe_distance
    (out/'sdk-config.json').write_text(json.dumps(config,indent=2))
    statuses=collections.Counter();count=0;began=time.monotonic();written=False
    trace=(out/'native-vio.jsonl').open('w')

    def vio(o):
        nonlocal count
        trace.write(o.asJson()+'\n');statuses[str(o.status)]+=1;count+=1
        if count%300==0:trace.flush();print(json.dumps({'stage':'sensor_pose','poses':count,'elapsed_s':time.monotonic()-began}),flush=True)

    def mapping(o):
        nonlocal written
        if not o.finalMap:
            # SDK image buffers are ephemeral. Copy pixels on arrival instead
            # of retaining native objects or assuming final-map bitmaps exist.
            for frame_id in o.updatedKeyFrames:
                path=cache/f'{frame_id}.png'
                kf=o.map.keyFrames.get(frame_id)
                if path.exists() or kf is None:continue
                frame=kf.frameSet.rgbFrame or kf.frameSet.primaryFrame
                if frame.image is not None:
                    rgb=np.copy(frame.image.toArray()).astype(np.uint8)
                    Image.fromarray(rgb[:,:,:3]).save(path)
            return
        cameras=[];all_observations=[];points={};colors=collections.defaultdict(list);residuals=[];nonpositive=0
        for ordinal,frame_id in enumerate(sorted(o.map.keyFrames)):
            kf=o.map.keyFrames[frame_id];fs=kf.frameSet;raw=fs.rgbFrame or fs.primaryFrame
            frame=raw
            rgb=np.array(Image.open(cache/f'{frame_id}.png').convert('RGB'))
            if rgb.shape[2]>3:rgb=rgb[:,:,:3]
            h,w=rgb.shape[:2];scale=a.width/h;target_h=round(w*scale)
            pose=frame.cameraPose.getCameraToWorldMatrix();K=frame.cameraPose.camera.getIntrinsicMatrix()
            stamp=frame.cameraPose.pose.time;source_index=int(np.argmin(abs(frame_times-stamp)))
            filename=f'frame-{source_index+1:05d}.jpg';train=(ordinal%7)!=3
            Image.fromarray(rgb).transpose(Image.Transpose.ROTATE_270).resize((a.width,target_h),Image.Resampling.LANCZOS).save(out/'dataset/images'/filename,quality=95)
            matrix=np.linalg.inv(pose);observations=[]
            for obs in frame.sparseFeatures:
                xyz=np.array([obs.position.x,obs.position.y,obs.position.z]);pixel=np.array([obs.pixelCoordinates.x,obs.pixelCoordinates.y])
                camera=matrix[:3,:3]@xyz+matrix[:3,3]
                if camera[2]>0:
                    projected=K@camera;residuals.append(float(np.linalg.norm(projected[:2]/projected[2]-pixel)))
                else:nonpositive+=1
                observations.append({'id':obs.id,'xy':pixel.tolist(),'xyz':xyz.tolist()})
                if train:
                    points[obs.id]=xyz
                    x=int(np.clip(round(pixel[0]),0,w-1));y=int(np.clip(round(pixel[1]),0,h-1));colors[obs.id].append(rgb[y,x].tolist())
            # Rotate stored pixels clockwise and rotate camera axes consistently.
            rotation=np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
            upright_pose=pose@rotation
            scaled=np.array([[K[1,1],0,h-1-K[1,2]],[0,K[0,0],K[0,2]],[0,0,1.]])
            scaled[:2]*=scale
            cameras.append({'id':frame_id,'source_frame_number':source_index,'source_timestamp':stamp,
                            'source_time_s':float(stamp-start),'file':filename,'train':train,
                            'camera_to_world_opencv':upright_pose.tolist(),'K':scaled.tolist(),'source_K':K.tolist(),
                            'source_camera_to_world_opencv':pose.tolist(),'pixel_rotation_clockwise_deg':90,
                            'source_resolution':[w,h],'resolution':[a.width,target_h]})
            all_observations.append({'id':frame_id,'file':filename,'features':observations})
        ids=sorted(points);xyz=np.array([points[i] for i in ids],dtype=np.float32)
        rgb=np.array([np.mean(colors[i],axis=0) for i in ids],dtype=np.uint8)
        poses=np.array([r['camera_to_world_opencv'] for r in cameras]);Ks=np.array([r['K'] for r in cameras])
        radius=float(np.linalg.norm(poses[:,:3,3]-np.median(poses[:,:3,3],axis=0),axis=1).max())
        np.savez_compressed(out/'dataset/scene.npz',points=xyz,colors=rgb,c2ws=poses,Ks=Ks,
                            train=np.array([r['train'] for r in cameras]),names=np.array([r['file'] for r in cameras]),scene_scale=radius)
        (out/'keyframes.json').write_text(json.dumps(cameras,indent=2))
        (out/'sparse-observations.json').write_text(json.dumps(all_observations))
        np.savez_compressed(out/'native-training-points.npz',ids=ids,xyz=xyz,rgb=rgb)
        report={'pose_source':('SpectacularAI1.54.0 upstream fast VIO/SLAM, final optimized keyframes' if a.fast else 'SpectacularAI1.54.0 offline visual-inertial SLAM, final optimized keyframes'),
                'camera_convention':'OpenCV x-right/y-down/z-forward camera-to-world; SDK world Z-up',
                'pixel_geometry':'Stored pixels rotated90deg clockwise; camera axes and pinhole K rotated consistently, then resized',
                'metric_scale':None,'sdk_estimated_units':'metres, not independently measured',
                'source_video_frames':len(source_frames),'modeled_keyframes':len(cameras),
                'training_images':sum(r['train'] for r in cameras),'heldout_images':sum(not r['train'] for r in cameras),
                'holdout_selection':'Ordinal3 modulo7 of final source-ordered keyframes; pose solve observes the whole capture',
                'initial_points':len(xyz),'initial_color_source':'Training-keyframe RGB observations only; no heldout colors',
                'reprojection_residual_px_min_p10_median_p90_max':np.quantile(residuals,[0,.1,.5,.9,1]).tolist(),
                'nonpositive_depth_observations':nonpositive,'source_time_s_first_last':[cameras[0]['source_time_s'],cameras[-1]['source_time_s']],
                'resolution':cameras[0]['resolution'],'scene_radius_estimated_units':radius,
                'limitations':['Pose-conditioned heldout RGB is correlated and not independent localization accuracy.',
                               'No surveyed scale/floor, measured depth, collision mesh, robot policy or dynamics.',
                               'Video renders follow final modeled keyframes and omit unmodeled startup intervals; timing is compressed.',
                               'SDK pose/geometry quality is an estimate; native reprojection residuals are evidence, not a promotion gate.']}
        (out/'scene.json').write_text(json.dumps(report,indent=2))
        (out/'dataset/scene-metadata.json').write_text(json.dumps(report,indent=2));written=True
        print(json.dumps({'stage':'sensor_map','report':report}),flush=True)

    try:
        replay=spectacularAI.Replay(str(a.input),mapperCallback=mapping,configuration=config,ignoreFolderConfiguration=True)
        replay.setOutputCallback(vio);replay.runReplay();replay.close()
    finally:trace.close()
    if not written:raise RuntimeError('SDK did not deliver a final map')
    (out/'sdk-completion.json').write_text(json.dumps({'native_pose_observations':count,'status_counts':dict(statuses),'elapsed_s':time.monotonic()-began,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
