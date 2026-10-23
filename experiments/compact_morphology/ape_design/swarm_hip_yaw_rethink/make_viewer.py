"""Export a lightweight 500 mm visual coupon from the source shell STL meshes."""
from pathlib import Path
import math
import trimesh

ROOT=Path(__file__).resolve().parents[4]
ART=ROOT/'artifacts/hip-yaw-rethink'; ART.mkdir(parents=True,exist_ok=True)
SHELL=ROOT/'artifacts/sketch-ape-shell/exports/500mm/stl'
K=500/350

def paint(mesh,color):
    mesh.visual.vertex_colors=color
    return mesh
def add(scene,name,mesh,color):
    scene.add_geometry(paint(mesh,color),geom_name=name,node_name=name)
def box(scene,name,center,size,color):
    m=trimesh.creation.box(extents=size);m.apply_translation(center);add(scene,name,m,color)
def cyl(scene,name,center,r,h,color):
    m=trimesh.creation.cylinder(radius=r,height=h,sections=40);m.apply_translation(center);add(scene,name,m,color)
def torus(scene,name,center,major=6.5,minor=1.75,color=(186,157,107,255)):
    m=trimesh.creation.torus(major_radius=major,minor_radius=minor,major_sections=40,minor_sections=12)
    m.apply_translation(center);add(scene,name,m,color)
def export(layout):
    scene=trimesh.Scene()
    for n in ('torso_front','torso_back','pelvis_front','pelvis_back'):
        mesh=trimesh.load(SHELL/f'{n}.stl',force='mesh')
        add(scene,'skin_'+n,mesh,(225,218,202,165))
    for side in (-1,1):
        x=side*26.5*K; z=160*K
        mx=x if layout=='coaxial' else side*13.5*K
        my=0 if layout=='coaxial' else -10*K
        cyl(scene,f'{layout}_RSF5B_{side:+}',(mx,my,z+17.9),12.5,35.8,(68,115,158,255))
        cyl(scene,f'{layout}_output_shaft_{side:+}',(x,0,z-8),3,16,(184,161,112,255))
        for dz in (-10,-2):torus(scene,f'{layout}_W618_bearing_{side:+}_{dz}',(x,0,z+dz))
        cyl(scene,f'{layout}_yaw_carrier_{side:+}',(x,0,z-14),5,4,(184,161,112,255))
        if layout=='remote':
            # Flat-belt/pulley diagram envelope; transfer hardware is unselected.
            dx=x-mx;dy=-my;span=math.hypot(dx,dy);angle=math.atan2(dy,dx)
            bar=trimesh.creation.box(extents=(span,16,2))
            bar.apply_transform(trimesh.transformations.rotation_matrix(angle,(0,0,1)))
            bar.apply_translation(((x+mx)/2,my/2,z-2))
            add(scene,f'remote_belt_clearance_envelope_{side:+}',bar,(206,163,83,235))
            cyl(scene,f'remote_input_pulley_{side:+}',(mx,my,z-2),5,4,(180,143,84,255))
            cyl(scene,f'remote_output_pulley_{side:+}',(x,0,z-2),5,4,(180,143,84,255))
    # Fixed lower bridge and output-bearing cheek supports.
    box(scene,'fixed_cross_bridge',(0,0,160*K-10),(66,8,3.5),(88,105,103,255))
    for side in (-1,1):box(scene,f'upper_bearing_cheek_{side:+}',(side*32*K,0,160*K-5),(4,8,8),(88,105,103,255))
    box(scene,'Tattu_4S850_source_z200',(0,0,200*K),(60,30,30.5),(221,145,56,235))
    box(scene,'shared_5V_12V_CAN_service_z225',(0,0,225*K),(25,20,14),(139,103,77,240))
    # Shoulder envelopes reproduce the separate team's latest 500 mm bounds.
    for side in (-1,1):
        box(scene,f'shoulder_yaw_case_{side:+}',(side*55,5,305.5),(20,34,29),(71,126,155,185))
        box(scene,f'shoulder_pitch_case_{side:+}',(side*55,-12,362.5),(20,34,34),(78,138,166,175))
    scene.export(ART/f'{layout}-500.glb')

if __name__=='__main__':
    export('coaxial');export('remote')
