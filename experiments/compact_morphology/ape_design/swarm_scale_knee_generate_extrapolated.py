"""Explicit 550/600 mm extrapolation of sketch_ape_shell Study into a unique artifact.

The source Study deliberately validates 350–500 mm. This subclass changes only
that range check; station geometry and the independent 2 mm output wall remain
the source generator's. It is a screen, not a validated source-shell release.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ORIGIN = ROOT/'experiments/sketch_ape_shell/build.py'
OUT = ROOT/'artifacts/swarm-scale-knee-20261004'
spec=importlib.util.spec_from_file_location('sketch_ape_source',ORIGIN)
assert spec and spec.loader
source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

class ExtrapolatedStudy(source.Study):
    def __init__(self,height):
        if height not in (550,600):
            raise ValueError('This screen only extrapolates 550 or 600 mm')
        self.height=height
        self.s=height/350
        self.wall=2.0
        self.t=self.wall/self.s
        self.gap=.65/self.s
        self.parts=[]

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('height',type=int,choices=(550,600))
    args=ap.parse_args()
    ExtrapolatedStudy(args.height).build().export(OUT/f'source-{args.height}',False)
