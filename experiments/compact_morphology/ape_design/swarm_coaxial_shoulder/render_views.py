"""Capture matched opaque 50 cm sketch/coupon views into ignored artifacts.

Serve the repository root first with ``python3 -m http.server 8765``.  The
camera and lighting are fixed in viewer.html for contour comparison.
"""
from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts/coaxial-shoulder/renders'
CHROME = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
BASE = ('http://127.0.0.1:8765/experiments/compact_morphology/'
        'ape_design/swarm_coaxial_shoulder/viewer.html')


def render(pair):
    variant, view, pose = pair
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT/(f'{variant}_{view}' + ('_pose_'+pose if pose!='down' else '') + '.png')
    out.unlink(missing_ok=True)
    cmd = [str(CHROME), '--headless=new', '--enable-unsafe-swiftshader',
           '--use-angle=swiftshader', '--disable-gpu-sandbox',
           '--no-first-run', '--no-default-browser-check',
           '--disable-background-networking', '--disable-component-update',
           f'--user-data-dir=/tmp/ape-coaxial-shoulder-{variant}-{view}-{pose}',
           '--virtual-time-budget=8000', '--window-size=1100,1100',
           f'--screenshot={out}',
           f'{BASE}?variant={variant}&view={view}&pose={pose}&clean=1']
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic()+75
        while time.monotonic() < deadline:
            if out.exists() and out.stat().st_size > 10000:
                with Image.open(out) as img:
                    if max(ImageStat.Stat(img.convert('RGB')).stddev) > 9:
                        return out.name, out.stat().st_size
            time.sleep(.5)
        raise RuntimeError(f'timed out or blank: {out}')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == '__main__':
    pairs = [(v,p,'down') for v in ('source','candidate')
             for p in ('front','side','back','iso')]
    pairs += [('candidate',p,pose) for pose in ('forward','out','floor_reach')
              for p in ('front','side','iso')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name,size in pool.map(render,pairs):
            print(name,size,flush=True)
