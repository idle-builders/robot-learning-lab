"""Replay a Spectacular recording locally and retain native estimated poses.

Requires the SDK on Linux x86-64; sensor-runtime.Dockerfile is a local runtime.
Core Replay API follows SpectacularAI/sdk-examples/python/oak/vio_replay.py.
Configuration follows SDK CLI auto_config('ios-tof', mono=True, icp=False).
No Gaussian training, network calls or metric qualification are performed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time

import imageio_ffmpeg
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fast', action='store_true', help='Use upstream CLI fast configuration for a preliminary path')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not shutil.which('ffmpeg'):
        bindir = Path('/tmp/spectacular-ffmpeg')
        bindir.mkdir(exist_ok=True)
        ffmpeg = bindir/'ffmpeg'
        if not ffmpeg.exists():
            ffmpeg.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ['PATH'] = str(bindir)+os.pathsep+os.environ['PATH']
    import spectacularAI
    config = {'maxMapSize': 0, 'useSlam': True, 'useStereo': False,
              'passthroughColorImages': True, 'keyframeDecisionDistanceThreshold': .1,
              'icpVoxelSize': .1,
              'parameterSets': ['wrapper-base', 'ios-tof'] if args.fast else ['wrapper-base', 'offline-base', 'ios-tof']}
    if not args.fast:
        config.update(maxKeypoints=1000, optimizerMaxIterations=30)
    provenance = {'configuration': config, 'sdk_version': importlib.metadata.version('spectacularAI'),
                  'platform': platform.platform(), 'architecture': platform.machine(),
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in args.input.iterdir() if p.is_file()},
                  'fast_preview': args.fast,
                  'scope': 'Estimated monocular visual-inertial camera trajectory; no ground truth, splat or robot policy.'}
    (args.output/'configuration.json').write_text(json.dumps(provenance, indent=2)+'\n')
    count = 0
    began = time.monotonic()
    stream = (args.output/'vio.jsonl').open('w')

    def on_vio(output):
        nonlocal count
        stream.write(output.asJson()+'\n')
        count += 1
        if count % 100 == 0:
            stream.flush()
            print(json.dumps({'observations': count, 'elapsed_s': time.monotonic()-began}), flush=True)

    def on_map(output):
        if not output.finalMap:
            return
        keyframes, points = [], {}
        for frame_id, keyframe in output.map.keyFrames.items():
            frame = keyframe.frameSet.rgbFrame or keyframe.frameSet.primaryFrame
            keyframes.append({'id': frame_id, 'camera_to_world': frame.cameraPose.getCameraToWorldMatrix().tolist()})
            for obs in frame.sparseFeatures:
                points[obs.id] = [obs.position.x, obs.position.y, obs.position.z]
        (args.output/'final-keyframes.json').write_text(json.dumps(keyframes, indent=2)+'\n')
        np.savez_compressed(args.output/'sparse-map.npz', ids=np.array(list(points)), xyz=np.array(list(points.values())))
        print(json.dumps({'final_keyframes': len(keyframes), 'sparse_points': len(points)}), flush=True)

    try:
        replay = spectacularAI.Replay(str(args.input), mapperCallback=on_map,
                                     configuration=config, ignoreFolderConfiguration=True)
        replay.setOutputCallback(on_vio)
        replay.runReplay()
        replay.close()
        (args.output/'completion.json').write_text(json.dumps({'observations': count, 'elapsed_s': time.monotonic()-began}, indent=2)+'\n')
    finally:
        stream.close()


if __name__ == '__main__':
    main()
