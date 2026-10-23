"""Capture matched opaque viewer views with local headless Chrome.

Start a localhost file server at the repository root first, e.g.
``python3 -m http.server 8765 --bind 127.0.0.1``.
The camera and lighting are locked by viewer.html, so contours compare directly.
"""
from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageStat

HERE = Path(__file__).resolve().parent
CHROME = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
BASE = ('http://127.0.0.1:8765/experiments/compact_morphology/'
        'ape_design/swarm_topology_wildcard_v2/viewer.html')


def render(pair):
    variant, view = pair
    out = HERE/'renders'/f'{variant}_{view}.png'
    out.parent.mkdir(exist_ok=True)
    cmd = [str(CHROME), '--headless=new', '--enable-unsafe-swiftshader',
           '--use-angle=swiftshader', '--disable-gpu-sandbox',
           '--no-first-run', '--no-default-browser-check',
           '--disable-background-networking', '--disable-component-update',
           f'--user-data-dir=/tmp/ape-sculpt-{variant}-{view}',
           '--virtual-time-budget=8000', '--window-size=1100,1100',
           f'--screenshot={out}',
           f'{BASE}?variant={variant}&view={view}&clean=1']
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+75
        while time.monotonic()<deadline:
            if out.exists() and out.stat().st_size>10000:
                # A background-only image is about 6 kB; check contrast too.
                with Image.open(out) as img:
                    s=ImageStat.Stat(img.convert('RGB'))
                    if max(s.stddev)>9:
                        return out.name,out.stat().st_size
            time.sleep(.5)
        raise RuntimeError(f'timed out/blank: {out}')
    finally:
        proc.terminate()
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:proc.kill()


if __name__ == '__main__':
    pairs=[(v,p) for v in ('source','visual','clearance')
           for p in ('front','side','back','iso')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name,size in pool.map(render,pairs):
            print(name,size,flush=True)
