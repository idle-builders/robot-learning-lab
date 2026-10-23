"""Representative full-body hard-case collision screen for the exposed ape frame.

This joins the upper and lower joint coupons in a few named postures. It is not
continuous clearance, a force check, or a print tolerance proof.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import build123d as bd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.compact_morphology.ape_frame50.upper_frame_candidate import build,pose
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import candidate,intersection
from experiments.compact_morphology.ape_frame50.shin_ankle_package import package,foot
from experiments.compact_morphology.ape_frame50.hip_knee_package import turn_knee_side
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip
from experiments.compact_morphology.ape_design.swarm_cove_hip import build as old

OUT=ROOT/'artifacts/ape-frame50-upper-frame/full-body-case-screen.json'
POSES={
    'stand':{'upper':(0,0,0,0),'hip':'stand','knee':0,'ankle':0},
    'step':{'upper':(25,0,-45,-50),'hip':'stride','knee':53,'ankle':0},
    'squat':{'upper':(0,0,-30,-60),'hip':'squat','knee':85,'ankle':0},
    'reach':{'upper':(25,-25,-65,-65),'hip':'stand','knee':0,'ankle':0},
    'crawl':{'upper':(0,-15,-75,-80),'hip':'crawl','knee':45,'ankle':0},
}


def bbox_overlap(a,b):
    aa,bb=a.bounding_box(),b.bounding_box()
    return all(getattr(aa.min,c)<getattr(bb.max,c)-0.05 and
               getattr(bb.min,c)<getattr(aa.max,c)-0.05 for c in 'XYZ')


def lower_shapes(hw,thigh,shin,ankle,foot_shapes,angles,knee):
    result=[]
    for side,sign in (('right',1),('left',-1)):
        for stage,group in hip.frame(hw,sign,*angles).items():
            result.extend((f'{side}_hip_{stage}_{n}',s) for n,s in group.items())
        for name,raw in thigh.items():
            shape=raw.mirror(bd.Plane.YZ) if sign==-1 else raw
            result.append((f'{side}_thigh_{name}',old.transform(shape,angles[0]*sign,angles[1],angles[2],'thigh',sign)))
        for prefix,group in (('shin',shin),('ankle',ankle),('foot',foot_shapes)):
            for name,raw in group.items():
                shape=turn_knee_side(raw,knee,sign)
                result.append((f'{side}_{prefix}_{name}',old.transform(shape,angles[0]*sign,angles[1],angles[2],'thigh',sign)))
    return result


def main():
    hw,_=candidate()
    thigh,shin,ankle=package()
    upper=build()
    reports={}
    for name,config in POSES.items():
        lower=lower_shapes(hw,thigh,shin,ankle,foot(config['ankle']),hip.POSES[config['hip']],config['knee'])
        upper_cases=[]
        upper_solids={"right":[],"left":[]}
        for side,sign in (('right',1),('left',-1)):
            for stage in ('fixed','yaw','pitch','arm','forearm'):
                for item,raw in upper[stage].items():
                    shape=pose(raw,stage,config['upper'])
                    if sign==-1:shape=shape.mirror(bd.Plane.YZ)
                    label=f'{side}_{stage}_{item}'
                    upper_solids[side].append((label,shape))
                    if 'case' in item:upper_cases.append((label,shape))
        hits=[]
        for label,case in upper_cases:
            for other,shape in lower:
                if not bbox_overlap(case,shape):continue
                volume=intersection(case,shape)
                if volume>0.05:hits.append([label,other,round(volume,3)])
        lower_cases=[(label,shape) for label,shape in lower if 'case' in label]
        stock_hits=[]
        for side in ('right','left'):
            for label,stock in upper_solids[side]:
                if 'case' in label or 'bearing' in label:continue
                for other,case in lower_cases:
                    if not bbox_overlap(stock,case):continue
                    volume=intersection(stock,case)
                    if volume>0.05:stock_hits.append([label,other,round(volume,3)])
        bilateral_hits=[]
        for label,right in upper_solids['right']:
            for other,left in upper_solids['left']:
                if not bbox_overlap(right,left):continue
                volume=intersection(right,left)
                if volume>0.05:bilateral_hits.append([label,other,round(volume,3)])
        reports[name]={'upper_case_vs_lower_hard_hits':hits,
                       'upper_stock_vs_lower_case_hard_hits':stock_hits,
                       'upper_right_vs_left_hard_hits':bilateral_hits,
                       'upper_cases':len(upper_cases),'lower_solids':len(lower)}
    result={'stage':'cross-body case screen, selected postures only; not continuous or toleranced',
            'poses':POSES,'results':reports,
            'open':['continuous joint-space and cable sweep','case mounting, drive transfer and load-bearing strength']}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({n:{key:len(v[key]) for key in ('upper_case_vs_lower_hard_hits',
          'upper_stock_vs_lower_case_hard_hits','upper_right_vs_left_hard_hits')}
          for n,v in reports.items()},indent=2))


if __name__=='__main__':main()
