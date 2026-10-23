"""Browser playback derivatives; originals and simulation evidence stay immutable.

The in-app browser advances H.264 playback without displaying its picture on this
machine. Native FFmpeg VP9/WebM renders correctly in the same Gradio player.
Gradio's format conversion writes beside its input, so use a sealed, hash-keyed
artifact instead of asking Gradio to convert an immutable experiment in place.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def browser_video(source):
    if source is None:
        return None
    source = Path(source).absolute()
    if not source.is_relative_to(ROOT / 'artifacts') or not source.is_file():
        raise ValueError('Playback requires a retained local artifact')
    if source.suffix.lower() == '.webm':
        return str(source)
    sha = _digest(source)
    prefix = 'browser-playback-' + sha
    for manifest in (ROOT / 'artifacts').glob(prefix + '*/playback.json'):
        data = json.loads(manifest.read_text())
        video = manifest.parent / 'replay.webm'
        if data['source_sha256'] == sha and video.is_file() and _digest(video) == data['video_sha256']:
            return str(video)

    import imageio_ffmpeg
    from lab.artifact_store import snapshot_file, plan_migration, apply_migration, verify_manifest

    out = ROOT / 'artifacts' / (prefix + '-' + uuid.uuid4().hex[:8])
    out.mkdir()
    snapshot_file(source, out / 'source' / ('input' + source.suffix))
    snapshot_file(Path(__file__), out / 'source/browser_media.py')
    video = out / 'replay.webm'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(source),
               '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '32', '-pix_fmt', 'yuv420p',
               '-c:a', 'libopus', str(video)]
    try:
        with (out / 'ffmpeg.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        (out / 'playback.json').write_text(json.dumps({
            'source': str(source.relative_to(ROOT)), 'source_sha256': sha,
            'video_sha256': _digest(video), 'command': command,
            'scope': 'VP9 playback derivative; original recording retained without modification',
        }, indent=2))
    finally:
        plan = plan_migration(ROOT, [out])
        apply_migration(ROOT, plan)
        verify_manifest(ROOT, plan)
    return str(video)
