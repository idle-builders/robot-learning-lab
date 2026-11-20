"""Inspect a Spectacular ZIP locally; no training, pose claims or provider calls.

Run with .venv-learning/bin/python. The original ZIP is the retained input.
Extracted inputs and review-sheet.jpg regenerate exactly with this invocation.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def quantiles(values):
    return np.quantile(values, [0, .1, .5, .9, 1]).tolist() if len(values) else []


def timing(times):
    times = np.asarray(times)
    dt = np.diff(times)
    return {'count': len(times), 'span_s': float(times[-1]-times[0]),
            'nonincreasing_intervals': int(np.sum(dt <= 0)),
            'interval_s_min_p10_median_p90_max': quantiles(dt),
            'mean_rate_hz': float((len(times)-1)/(times[-1]-times[0]))}


def matches(a, b, orb):
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return {'features_a': len(ka), 'features_b': len(kb), 'ratio_matches': 0, 'homography_inliers': 0}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .75*pair[1].distance]
    inliers = 0
    if len(good) >= 4:
        _, mask = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
                                     np.float32([kb[m.trainIdx].pt for m in good]), cv2.RANSAC, 3)
        if mask is not None:
            inliers = int(mask.sum())
    return {'features_a': len(ka), 'features_b': len(kb), 'ratio_matches': len(good),
            'homography_inliers': inliers}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--review-rotate-clockwise', type=int, choices=[0, 90, 180, 270], default=0,
                    help='Review-sheet display only; keep native video and calibration unchanged')
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    inputs = out/'input'
    inputs.mkdir(exist_ok=True)
    inventory = []
    with zipfile.ZipFile(args.zip) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            member = Path(item.filename)
            if member.is_absolute() or '..' in member.parts:
                raise ValueError('Unsafe archive member')
            path = inputs/member
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as src, path.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            inventory.append({'member': item.filename, 'path': str(path), 'bytes': item.file_size,
                              'sha256': digest(path)})
    data_path = next(inputs.rglob('data.jsonl'))
    folder = data_path.parent
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    frames = [r for r in rows if 'frames' in r]
    frame_times = np.array([r['time'] for r in frames])
    relative_times = frame_times-frame_times[0]
    sensors = collections.defaultdict(list)
    for row in rows:
        if 'sensor' in row:
            sensors[row['sensor']['type']].append(row)
    calibration = json.loads((folder/'calibration.json').read_text())
    metadata = json.loads((folder/'metadata.json').read_text())
    sensor_reports = {}
    for kind, values in sensors.items():
        times = np.array([r['time'] for r in values])
        ordered = np.sort(times)
        ix = np.searchsorted(ordered, frame_times).clip(1, len(ordered)-1)
        nearest = np.minimum(abs(frame_times-ordered[ix-1]), abs(frame_times-ordered[ix]))
        sensor_reports[kind] = {**timing(times),
            'vector_norm_min_p10_median_p90_max': quantiles(np.linalg.norm([r['sensor']['values'] for r in values], axis=1)),
            'frame_nearest_sample_distance_ms_min_p10_median_p90_max': quantiles(nearest*1000),
            'frames_before_first_sample': int(np.sum(frame_times < times.min())),
            'frames_after_last_sample': int(np.sum(frame_times > times.max()))}
    per_frame = [r['frames'][0] for r in frames]
    intrinsics = {key: quantiles([r['calibration'][key] for r in per_frame if 'calibration' in r])
                  for key in ['focalLengthX', 'focalLengthY', 'principalPointX', 'principalPointY']}
    video = next(p for p in folder.glob('data.*') if p.suffix.lower() in ['.mov', '.mp4', '.mkv'])
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    indices = np.linspace(0, len(frames)-1, 24).round().astype(int)
    tiles = {}
    sampled, pairs = [], []
    orb = cv2.ORB_create(1600)
    cv2.setRNGSeed(0)
    first_grey = previous_grey = last_grey = None
    previous_time = None
    count = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        time_s = float(relative_times[count]) if count < len(frames) else count/fps
        if count in indices:
            tiles[count] = (time_s, cv2.cvtColor(cv2.resize(image, (320, 240)), cv2.COLOR_BGR2RGB))
        if count % max(1, round(fps/2)) == 0:
            grey = cv2.cvtColor(cv2.resize(image, (1280, round(height*1280/width))), cv2.COLOR_BGR2GRAY)
            sampled.append({'frame': count, 'time_s': time_s,
                            'laplacian_variance_grey8': float(cv2.Laplacian(grey, cv2.CV_64F).var()),
                            'mean_luminance_grey8': float(grey.mean()),
                            'near_black_fraction_le3': float((grey <= 3).mean()),
                            'near_white_fraction_ge252': float((grey >= 252).mean())})
        if count % max(1, round(fps)) == 0:
            grey = cv2.cvtColor(cv2.resize(image, (960, round(height*960/width))), cv2.COLOR_BGR2GRAY)
            if first_grey is None:
                first_grey = grey
            if previous_grey is not None:
                pairs.append({'start_s': previous_time, 'end_s': time_s, **matches(previous_grey, grey, orb)})
            previous_grey, previous_time, last_grey = grey, time_s, grey
        count += 1
    cap.release()
    tile_width, tile_height = (240, 320) if args.review_rotate_clockwise in [90, 270] else (320, 240)
    sheet = Image.new('RGB', (4*tile_width, 6*(tile_height+30)), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
    for i, index in enumerate(indices):
        if index not in tiles:
            continue
        t, pixels = tiles[index]
        x, y = (i % 4)*tile_width, (i//4)*(tile_height+30)
        tile = Image.fromarray(pixels).rotate(-args.review_rotate_clockwise, expand=True)
        sheet.paste(tile, (x, y))
        draw.text((x+5, y+tile_height+3), f'{t:.1f} s · frame {index}', fill='black', font=font)
    sheet.save(out/'review-sheet.jpg', quality=93)
    (out/'frame-measurements.json').write_text(json.dumps(sampled, indent=2)+'\n')
    (out/'adjacent-view-measurements.json').write_text(json.dumps(pairs, indent=2)+'\n')
    report = {
        'schema': 'spectacular-capture-audit.v1', 'archive': str(args.zip),
        'archive_sha256': digest(args.zip), 'archive_bytes': args.zip.stat().st_size,
        'members': inventory, 'metadata': metadata, 'calibration': calibration,
        'frame_timing': timing(frame_times), 'video': {'width': width, 'height': height,
            'reported_frames': reported_count, 'decoded_frames': count, 'reported_fps': fps,
            'frame_record_count_difference': count-len(frames)},
        'frame_number_differences': dict(collections.Counter(np.diff([r['number'] for r in frames]).tolist())),
        'per_frame_intrinsics_min_p10_median_p90_max': intrinsics,
        'exposure_seconds_min_p10_median_p90_max': quantiles([r['exposureTimeSeconds'] for r in per_frame]),
        'sensors': sensor_reports, 'saved_pose_fields': sorted({k for r in rows for k in r if k not in ['time','sensor','number','frames']}),
        'image_measurements': {key: quantiles([r[key] for r in sampled]) for key in sampled[0] if key not in ['frame','time_s']},
        'adjacent_1s_view_measurements': {key: quantiles([r[key] for r in pairs]) for key in ['ratio_matches','homography_inliers']},
        'first_last_sample_feature_comparison': matches(first_grey, last_grey, orb),
        'source_sha256': digest(Path(__file__)),
        'review_rotation_clockwise_degrees': args.review_rotate_clockwise,
        'regenerate': f'.venv-learning/bin/python {Path(__file__).as_posix()} --zip {args.zip} --output {out} --review-rotate-clockwise {args.review_rotate_clockwise}',
        'limits': ['Same-clock timestamp proximity does not independently verify physical camera/IMU synchronization.',
                   'Homography feature support is an overlap diagnostic, not a 6-DoF pose, scale or loop-closure result.',
                   'Laplacian variance mixes texture and blur; inspect actual images.',
                   'Full video decoded; image statistics sampled at 2fps and feature pairs about 1s apart.',
                   'No automatic frame exclusions, no Gaussian fit, no Runpod requests.']}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['frame_timing','video','sensors','image_measurements','adjacent_1s_view_measurements','first_last_sample_feature_comparison']}, indent=2))


if __name__ == '__main__':
    main()
