"""Fetch one public ScanSceneverse GLB from its sorted tar shard by HTTP range.

The catalog's thumbnails and metadata are not scene qualification. This only
retrieves source geometry for a separate low-camera and floor-support screen.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import requests


DATA = Path('assets/real_indoor_scenes/scansceneverse')
BASE = 'https://huggingface.co/datasets/Stable-X/ScanSceneverse/resolve/main/shards/glbs'


def range_read(session: requests.Session, url: str, first: int, last: int) -> bytes:
    for attempt in range(4):
        try:
            response = session.get(url, headers={'Range': f'bytes={first}-{last}'},
                                   timeout=(15, 120))
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f'Server ignored range request: {response.status_code}')
            if len(response.content) != last - first + 1:
                raise IOError('Incomplete tar range response')
            return response.content
        except (requests.RequestException, IOError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError('unreachable')


def tar_header(block: bytes) -> tuple[str, int] | None:
    if len(block) != 512 or block[257:262] != b'ustar':
        return None
    expected = int(block[148:156].strip(b'\0 ') or b'0', 8)
    actual = sum(block[:148]) + 8 * 32 + sum(block[156:])
    if actual != expected:
        return None
    return block[:100].split(b'\0')[0].decode(), int(block[124:136].strip(b'\0 '), 8)


def next_header(session: requests.Session, url: str, offset: int, total: int) -> tuple[int, str, int] | None:
    chunk_size = 4 * 1024 * 1024
    while offset + 512 <= total:
        data = range_read(session, url, offset, min(total - 1, offset + chunk_size - 1))
        for index in range(0, len(data) - 511, 512):
            parsed = tar_header(data[index:index + 512])
            if parsed is not None:
                return offset + index, *parsed
        offset += len(data) // 512 * 512
    return None


def locate_fast(session: requests.Session, url: str, target: str,
                progress_path: Path | None = None) -> tuple[int, int]:
    response = session.get(url, headers={'Range': 'bytes=0-511'}, timeout=(15, 120))
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError('Server ignored tar range request')
    total = int(re.search(r'/([0-9]+)$', response.headers['Content-Range']).group(1))
    progress = (json.loads(progress_path.read_text()) if
                progress_path is not None and progress_path.is_file() else {})
    if (progress.get('url'), progress.get('target'), progress.get('total')) == \
            (url, target, total):
        low, high = int(progress['low']), int(progress['high'])
        completed = int(progress['completed_iterations'])
    else:
        low, high, completed = 0, total, 0
    for iteration in range(completed, 35):
        if high - low < 512:
            break
        if high - low <= 4 * 1024 * 1024:
            # `low` is a tar-header boundary. Once the name bracket is small,
            # skip members by their recorded sizes rather than downloading a
            # multi-megabyte window for every remaining binary-search step.
            cursor = low
            while cursor <= high:
                parsed = tar_header(range_read(session, url, cursor, cursor + 511))
                if parsed is None:
                    break
                name, size = parsed
                if name == target:
                    if progress_path is not None:
                        progress_path.unlink(missing_ok=True)
                    return cursor, size
                if name > target:
                    break
                cursor += 512 + math.ceil(size / 512) * 512
            raise FileNotFoundError(target)
        mid = ((low + high) // 1024) * 512
        found = next_header(session, url, mid, total)
        if found is None:
            high = mid
        else:
            offset, name, size = found
            if name == target:
                if progress_path is not None:
                    progress_path.unlink(missing_ok=True)
                return offset, size
            if name < target:
                low = offset + 512 + math.ceil(size / 512) * 512
            else:
                high = mid
        if progress_path is not None:
            progress_path.write_text(json.dumps({
                'url': url, 'target': target, 'total': total,
                'low': low, 'high': high,
                'completed_iterations': iteration + 1}) + '\n')
    found = next_header(session, url, low, total)
    if found is not None and found[1] == target:
        if progress_path is not None:
            progress_path.unlink(missing_ok=True)
        return found[0], found[2]
    raise FileNotFoundError(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('scene_id')
    parser.add_argument('--max-mb', type=float, default=300)
    parser.add_argument('--fast', action='store_true', help='Locate tar member by remote byte search')
    args = parser.parse_args()
    metadata = pd.read_parquet(DATA / 'metadata.parquet').set_index('scene_id')
    if args.scene_id not in metadata.index:
        raise ValueError(f'Unknown scene: {args.scene_id}')
    record = metadata.loc[args.scene_id]
    if record['license_slug'] not in {'by', 'by-sa', 'by-nc', 'by-nc-sa', 'cc0'}:
        raise ValueError(f'Unsupported license: {record["license_slug"]}')
    target = record['file_name']
    starts = json.loads((DATA / 'preview80/shard_starts.json').read_text())
    shard = bisect.bisect_right(starts, target) - 1
    if shard < 0:
        raise ValueError('Target precedes first shard')
    url = f'{BASE}/glbs-{shard:04d}.tar'
    session = requests.Session()
    offset = 0
    for index in range(2000):
        if args.fast:
            progress_path = DATA / 'glbs' / f'{args.scene_id}.locate-progress.json'
            location_path = DATA / 'glbs' / f'{args.scene_id}.tar-location.json'
            previous = (json.loads(location_path.read_text()) if
                        location_path.is_file() else {})
            if (previous.get('url'), previous.get('target')) == (url, target):
                offset, size = int(previous['offset']), int(previous['size'])
            else:
                offset, size = locate_fast(session, url, target, progress_path)
                location_path.write_text(json.dumps({
                    'url': url, 'target': target,
                    'offset': offset, 'size': size}) + '\n')
            name = target
        else:
            header = range_read(session, url, offset, offset + 511)
            parsed = tar_header(header)
            if parsed is None:
                raise FileNotFoundError(target)
            name, size = parsed
        if name == target:
            if size > args.max_mb * 1_000_000:
                raise ValueError(f'{target} is {size/1e6:.1f} MB, over size limit')
            output = DATA / 'glbs' / f'{args.scene_id}.glb'
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_suffix('.glb.part')
            if partial.is_file() and partial.stat().st_size > size:
                partial.unlink()
            written = partial.stat().st_size if partial.is_file() else 0
            with partial.open('ab') as destination:
                while written < size:
                    first = offset + 512 + written
                    last = min(offset + 511 + size, first + 4 * 1024 * 1024 - 1)
                    block = range_read(session, url, first, last)
                    destination.write(block)
                    written += len(block)
            if partial.stat().st_size != size:
                raise IOError('Incomplete GLB download')
            digest = hashlib.sha256()
            with partial.open('rb') as source_file:
                for block in iter(lambda: source_file.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            sha = digest.hexdigest()
            partial.replace(output)
            sidecar = output.with_suffix('.source.json')
            sidecar.write_text(json.dumps({
                'scene_id': args.scene_id, 'name': record['name'],
                'source_url': record['source_url'], 'license': record['license_slug'],
                'license_url': record['license_url'], 'dataset_url': url,
                'author': record['author_display_name'],
                'author_username': record['author_username'],
                'tar_member': name, 'tar_member_index': None if args.fast else index,
                'tar_header_offset': offset, 'size_bytes': size,
                'sha256': sha,
            }, indent=2) + '\n')
            print(output)
            return
        if name > target:
            raise FileNotFoundError(target)
        offset += 512 + math.ceil(size / 512) * 512
    raise RuntimeError('Tar shard exceeded 2000 members')


if __name__ == '__main__':
    main()
