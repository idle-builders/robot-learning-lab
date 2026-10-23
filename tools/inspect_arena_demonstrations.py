"""Inspect the pinned tutorial data with its native GR00T loader and Rerun.

This is a viewer adapter, not a dataset converter, policy or training loop.
Run with .venv-lerobot/bin/python; the original v2 data stays unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / '.cache/upstream/IsaacLab-Arena'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    args = parser.parse_args()
    pins = json.loads((ROOT/'workers/arena/pins.json').read_text())
    for repo, revision in [(ARENA, pins['arena']), (ARENA/'submodules/Isaac-GR00T', pins['gr00t'])]:
        assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() == revision
    sys.path[:0] = [str(ROOT), str(ARENA/'submodules/Isaac-GR00T'),
                   str(ARENA/'isaaclab_arena_gr00t/embodiments/g1')]
    import numpy as np
    import pandas as pd
    import rerun as rr
    import rerun.blueprint as rrb
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from g1_sim_wbc_data_config import unitree_g1_sim_wbc_config
    from lab.artifact_store import snapshot_file
    from tools.probe_vla import finalize

    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_file(Path(__file__), args.output/'source/inspect_arena_demonstrations.py')
    loader = LeRobotEpisodeLoader(args.dataset, unitree_g1_sim_wbc_config, video_backend='opencv')
    print('Loading native GR00T episode', args.episode, flush=True)
    episode = loader[args.episode]
    raw = pd.read_parquet(args.dataset/loader.data_path_pattern.format(
        episode_chunk=args.episode//loader.chunk_size, episode_index=args.episode))
    times = raw['timestamp'].to_numpy()
    assert len(episode) == len(raw) and np.isfinite(times).all() and (np.diff(times)>0).all()
    columns = {}
    for name in episode.columns:
        if name.startswith(('state.', 'action.')):
            values = np.stack(episode[name].to_numpy())
            assert len(values) == len(times) and np.isfinite(values).all()
            columns[name] = values
    images = np.stack(episode['video.ego_view'].to_numpy())
    assert images.shape == (len(times),480,640,3) and images.dtype == np.uint8
    rr.init('Arena G1 reference demonstrations')
    rr.save(str(args.output/'demonstration.rrd'))
    instruction = episode.iloc[0]['language.annotation.human.task_description']
    rr.log('description', rr.TextDocument(
        '# Published G1 demonstration\n\n'+instruction+
        '\n\nMimic-generated training example, not a new policy evaluation. '
        'Native GR00T loader; original data unchanged.'), static=True)
    for name, values in columns.items():
        kind, group = name.split('.',1)
        for axis in range(values.shape[1]):
            rr.send_columns(f'{kind}/{group}/{axis}', indexes=[rr.TimeColumn('time',duration=times)],
                            columns=rr.Scalars.columns(scalars=values[:,axis]))
    video = args.dataset/loader.video_path_pattern.format(
        episode_chunk=args.episode//loader.chunk_size,
        video_key='observation.images.ego_view', episode_index=args.episode)
    asset = rr.AssetVideo(path=video)
    video_times = asset.read_frame_timestamps_nanos()
    assert len(video_times) == len(times)
    rr.log('camera', asset, static=True)
    rr.send_columns('camera', indexes=[rr.TimeColumn('time',duration=times)],
                    columns=rr.VideoFrameReference.columns(timestamp=video_times))
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(rrb.Spatial2DView(origin='camera'), rrb.TextDocumentView(origin='description')),
        rrb.Horizontal(rrb.TimeSeriesView(name='Recorded state',origin='state'),
                       rrb.TimeSeriesView(name='Demonstrated actions',origin='action')))))
    rr.disconnect()
    report = {'status':'pass', 'scope':'Native dataset loading and demonstration inspection; no training.',
              'pins':pins, 'episodes':len(loader), 'episode_index':args.episode,
              'frames':len(episode), 'image_shape':list(images.shape),
              'language_instruction':instruction,
              'columns':{name:list(values.shape) for name,values in columns.items()},
              'video_frames':len(video_times),'time_range_s':[float(times[0]),float(times[-1])],
              'rerun_version':rr.__version__,
              'dataset_metadata':{name:{'sha256':hashlib.sha256((args.dataset/name).read_bytes()).hexdigest(),
                                        'content':(args.dataset/name).read_text()}
                                  for name in ['meta/info.json','meta/modality.json','meta/tasks.jsonl','meta/episodes.jsonl']}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    finalize(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
