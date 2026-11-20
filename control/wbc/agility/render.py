"""Kinematic playback clips of the agility motion set, one clip per family with its tempos side by side.

These show the references themselves (what the WBC would be asked to follow),
not a controller. Shorter tempos hold their last pose to the longest one.

    .venv-control/bin/python control/wbc/agility/render.py --motions DIR --output DIR
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'compute'))
import report  # noqa: E402  (WBC clip renderer: R3 panels side by side)

FAMILIES = ('chest beat', 'overhead wave', 'arm display swing', 'knee bounce', 'side sway', 'quick crouch', 'hops',
    'turn in place', 'march in place', 'retained forward-walk')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    catalogue = json.loads((args.motions / 'motions.json').read_text())['motions']
    args.output.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        rows = sorted((m for m in catalogue if m['family'] == family), key=lambda m: m['tempo'])
        if not rows:
            continue
        panels = []
        for m in rows:
            mo = np.load(args.motions / m['motion'])
            q = mo['qpos'][::int(mo['fps'][0]) // 50]
            panels.append((f"{m['tempo']:g} {m['tempo_unit']}", q))
        T = max(len(q) for _, q in panels)
        panels = [(label, np.concatenate([q, np.repeat(q[-1:], T - len(q), 0)])) for label, q in panels]
        path = args.output / f"{family.replace(' ', '-')}.mp4"
        report.clip(path, panels, family)
        print(path)


if __name__ == '__main__':
    main()
