"""Render full bounded diagnostic episodes from retained native states."""
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from common import *

def main():
    report=[]
    for episode in read(OUT/'replay-summary.json')['episodes']:
        model=mujoco.MjModel.from_xml_path(str(ROOT/episode['body']['path']));data=mujoco.MjData(model)
        trace=np.load(ROOT/episode['trace']['path']);renderer=mujoco.Renderer(model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.08,0,.23];camera.distance=1.0;camera.azimuth=140;camera.elevation=-15
        opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,0,0,0]
        path=OUT/'replay'/f'{episode["id"]}.mp4';times=np.arange(0,episode['actual_duration_s']-1e-9,1/30)
        writer=imageio.get_writer(path,fps=30,codec='libx264',quality=8,macro_block_size=None)
        for t in times:
            i=min(len(trace['time_s'])-1,int(np.searchsorted(trace['time_s'],t)))
            data.qpos[:]=trace['qpos'][i];data.qvel[:]=trace['qvel'][i];data.time=float(trace['time_s'][i]);mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera,scene_option=opt);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,43),fill=(14,20,25));draw.text((8,5),episode['id']+f' | t={data.time:.3f}s',fill='white')
            draw.text((8,22),'Prescribed old trajectory on new CAD body; no new policy training',fill=(235,197,133))
            writer.append_data(np.asarray(frame))
            if t==0 and episode['id']=='rigid-nominal-start0':frame.save(OUT/'model-replay-start.png')
        writer.close();renderer.close()
        reader=imageio.get_reader(path);count=0;std=[]
        for frame in reader:count+=1;std.append(float(frame.std()))
        reader.close()
        report.append(dict(episode=episode['id'],video=ref(path),trace=episode['trace'],body=episode['body'],
            expected_frames=len(times),decoded_frames=count,frame_size=[640,480],fps=30,
            minimum_frame_pixel_std=min(std),native_horizon_s=episode['actual_duration_s'],
            method='Later native state rendering of every30Hz video station through requested0.8s; no new dynamics or actor inference'))
        write(OUT/'video-verification.json',dict(clips=report));print(episode['id'],count,flush=True)
    # A plain neutral frame complements the diagnostic motion, with no source
    # trajectory pose masquerading as CAD neutral.
    model=mujoco.MjModel.from_xml_path(str(OUT/'cove50-rigid.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=600,width=800);camera.lookat[:]=[0,0,.26];camera.distance=.9
    renderer.update_scene(data,camera,scene_option=opt);Image.fromarray(renderer.render()).save(OUT/'model-neutral.png');renderer.close()

if __name__=='__main__':main()
