"""Exact 350 mm source-shell check of the lower z=200 pack reservation."""
import json
import build123d as b
import hip_yaw_rethink as h
from pathlib import Path

OUT=h.OUT/'battery-alternatives.json'
def box(c,d):return b.Pos(*c)*b.Box(*d)
rows=[]
for wall in (1,2):
    k,named,outer=h.shell(350,wall)
    skins=list(named.values())
    pack=box((0,0,200),(60,30,30.5))
    service=box((0,0,220),(25,20,14))
    shoulder=[]
    for side in (-1,1):
        shoulder.append((f'{side:+} shoulder yaw',box((side*55*.7,5*.7,305.5*.7),(20,34,29))))
        shoulder.append((f'{side:+} shoulder pitch',box((side*55*.7,-12*.7,362.5*.7),(20,34,34))))
    shellpack=[{'skin':name,'mm3':h.overlaps(pack,s)} for name,s in named.items() if h.overlaps(pack,s)>0.001]
    shellservice=[{'skin':name,'mm3':h.overlaps(service,s)} for name,s in named.items() if h.overlaps(service,s)>0.001]
    shoulderpack=[{'case':n,'mm3':h.overlaps(pack,s)} for n,s in shoulder if h.overlaps(pack,s)>0.001]
    motorrows=[]
    for layout in ('coaxial','remote'):
        for side in (-1,1):
            x=side*(h.HIP_X if layout=='coaxial' else 13.5)
            motor=h.rsf((x,0,h.HIP_Z),1)
            v=h.overlaps(pack,motor)
            if v>0.001:motorrows.append({'layout':layout,'side':side,'mm3':v})
    rows.append({'height_mm':350,'wall_mm':wall,'pack_center_source_mm':[0,0,200],
      'pack_size_mm':[60,30,30.5],'exact_pack_shell_hits':shellpack,'exact_service_shell_hits':shellservice,
      'exact_pack_service_mm3':h.overlaps(pack,service),'pack_shoulder_hits':shoulderpack,'pack_motor_hits':motorrows,
      'service_shoulder_hits':[{'case':n,'mm3':h.overlaps(service,s)} for n,s in shoulder if h.overlaps(service,s)>0.001]})
OUT.write_text(json.dumps({'source':'Study(350, wall, .65) exact shell BReps','rows':rows},indent=2))
for r in rows:print(r['wall_mm'],'pack shell',r['exact_pack_shell_hits'],'service shell',r['exact_service_shell_hits'],'yaw',r['pack_shoulder_hits'],'motor',r['pack_motor_hits'])
