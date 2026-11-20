"""RGB-only Gaussian observations in explicit OpenCV reconstruction coordinates.

Geometry-dependent consumers must inspect capabilities. A PLY or reconstructed
camera transform does not provide measured depth, metric pose or collisions.
"""
import base64
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import urllib.parse

import numpy as np
from PIL import Image


def consumer_geometry(manifest):
    width,height=manifest.get('image_size',[960,540])
    target_w,target_h=manifest.get('consumer_image_size',[640,360])
    if manifest.get('consumer_resize')=='letterbox':
        scale=min(target_w/width,target_h/height)
        w,h=round(width*scale),round(height*scale)
        return w/width,h/height,(target_w-w)//2,(target_h-h)//2,w,h
    return target_w/width,target_h/height,0,0,target_w,target_h


def source_rgb(image,manifest):
    _,_,x,y,w,h=consumer_geometry(manifest)
    result=Image.new('RGB',tuple(manifest.get('consumer_image_size',[640,360])))
    result.paste(image.convert('RGB').resize((w,h),Image.Resampling.BOX),(x,y))
    return np.array(result)


class GaussianRGBWorld:
    def __init__(self, manifest: Path, *, dashboard='http://127.0.0.1:8805'):
        self.manifest=json.loads(manifest.read_text())
        version=self.manifest['selected_visual_version']
        self.model=next(x for x in self.manifest['visual_model_candidates'] if x['version']==version)
        self.path=np.load(self.model['camera_path'])
        self.capabilities={'rgb':True,'depth_m':False,'metric_pose':False,'range_m':False,
                           'floor_alignment':False,'collision_geometry':False}
        self.worker=subprocess.Popen(['node',str(Path(__file__).with_name('browser_observer.cjs'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,start_new_session=True)
        self.runtime=self._request({'op':'init','dashboard':dashboard,
            'width':self.manifest.get('consumer_image_size',[640,360])[0],
            'height':self.manifest.get('consumer_image_size',[640,360])[1],
            'model_url':dashboard+'/asset?path='+urllib.parse.quote(self.model['gaussians'],safe='')})

    def _request(self, command):
        self.worker.stdin.write(json.dumps(command)+'\n');self.worker.stdin.flush()
        if not select.select([self.worker.stdout],[],[],45)[0]:
            self.close();raise TimeoutError('Gaussian observation exceeded 45 seconds')
        line=self.worker.stdout.readline()
        if not line: raise RuntimeError('Gaussian observation process ended before returning pixels')
        return json.loads(line)

    def camera(self, frame_name):
        index=self.path['names'].tolist().index(frame_name)
        k=self.path['Ks'][index].copy()
        sx,sy,x,y,_,_=consumer_geometry(self.manifest)
        k[0]*=sx;k[1]*=sy;k[0,2]+=x;k[1,2]+=y
        return self.path['c2ws'][index].copy(),k

    def observe(self, c2w, K, *, capture_time_s):
        response=self._request({'op':'frame','c2w':np.asarray(c2w).tolist(),'K':np.asarray(K).tolist()})
        rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(response['png_base64']))).convert('RGB'))
        if self.manifest.get('consumer_resize')=='letterbox':
            _,_,x,y,w,h=consumer_geometry(self.manifest)
            masked=np.zeros_like(rgb);masked[y:y+h,x:x+w]=rgb[y:y+h,x:x+w];rgb=masked
        return {'rgb':rgb,'capture_time_s':float(capture_time_s),'intrinsics':np.asarray(K).copy(),
                'rgb_source':'rendered_gaussian','depth_m':None,'metric_camera_pose':None,
                'render_sort':{key:response.get(key) for key in ['splats_sorted','splats_visible']},
                'render_pose_source':'evaluator_requested_'+self.manifest.get('pose_source','COLMAP_reconstruction_pose'),
                'capabilities':dict(self.capabilities)}

    def require_metric_navigation(self):
        missing=[k for k in ['depth_m','metric_pose','range_m','collision_geometry'] if not self.capabilities[k]]
        raise ValueError('Metric navigation unavailable; missing '+', '.join(missing))

    def close(self):
        if self.worker.poll() is None:
            self.worker.stdin.write('{"op":"close"}\n');self.worker.stdin.flush()
            self.worker.stdin.close()
            try:self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.worker.pid,signal.SIGTERM)
                try:self.worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.worker.pid,signal.SIGKILL);self.worker.wait()
