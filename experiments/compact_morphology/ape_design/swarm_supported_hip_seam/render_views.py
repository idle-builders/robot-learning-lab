"""Matched opaque source/candidate orthographic views from exported BRep meshes."""
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
VIEWS = {'front':(0,-1,.18),'side':(1,0,.1),'rear':(0,1,.18),'three-quarter':(1,-1,.35)}
POSES = ('stand','yaw_out','stride','squat')
SIZE = (440,460)

def render(path, view):
    scene=trimesh.load(path,force='scene')
    forward=np.array(VIEWS[view],dtype=float); forward/=np.linalg.norm(forward)
    up=np.array((0.,0.,1.)); right=np.cross(up,forward); right/=np.linalg.norm(right)
    up=np.cross(forward,right)
    verts=np.concatenate([g.vertices for g in scene.geometry.values()])
    center=np.array((29.,0.,250.))
    points=np.stack(((verts-center)@right,(verts-center)@up),axis=1)
    extent=max(np.ptp(points[:,0]),np.ptp(points[:,1]),160)
    scale=min(SIZE)*.79/extent
    img=Image.new('RGB',SIZE,(237,236,231));draw=ImageDraw.Draw(img)
    polygons=[];light=np.array((.4,-.5,.75));light/=np.linalg.norm(light)
    for geom in scene.geometry.values():
        v=geom.vertices
        q=np.stack(((v-center)@right,(v-center)@up),axis=1)*scale + np.array((SIZE[0]/2,SIZE[1]/2))
        q[:,1]=SIZE[1]-q[:,1]
        depth=(v-center)@forward
        for i,face in enumerate(geom.faces):
            shade=.78+.2*abs(float(geom.face_normals[i]@light))
            base=np.array((226,218,201))
            polygons.append((float(np.mean(depth[face])),tuple(map(tuple,q[face])),
                             tuple(np.clip(base*shade,0,255).astype(int))))
    for _,poly,color in sorted(polygons,key=lambda p:p[0]):draw.polygon(poly,fill=color)
    return img

def main():
    sheet=Image.new('RGB',(SIZE[0]*8,SIZE[1]*len(POSES)),'white')
    for i,pose in enumerate(POSES):
        for j,view in enumerate(VIEWS):
            for k,layer in enumerate(('source','candidate')):
                img=render(HERE/f'{layer}-{pose}.glb',view)
                ImageDraw.Draw(img).text((12,10),f'{pose} · {view} · {layer}',fill=(25,31,31))
                sheet.paste(img,((2*j+k)*SIZE[0],i*SIZE[1]))
    sheet.save(HERE/'matched-opaque-sheet.png')

if __name__=='__main__':main()
