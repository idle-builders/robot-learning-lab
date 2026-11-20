"""Small fresh beta-only posture-trajectory student, no state/critic/table query."""
from pathlib import Path
import json,time
import numpy as np
import torch
from adapter import ROOT,WalkingInterface,ref
from run import HERE,write
from collect_crouch import DATA as TEACHER_DATA,OUT as TEACHERS

DATA=ROOT/'data/basic-skills-retry-20261027/squat/command-student'
OUT=ROOT/'artifacts/cove50-basic-skills-retry/squat-20261027/command-student'


def model():return torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,22))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(138700)
    source=json.loads((TEACHER_DATA/'method.json').read_text());families=source['frozen_data_split'];xs={};ys={};rows=[]
    for family in sum(families.values(),[]):
        trace=TEACHERS/f'live-interface-com-gravity-reference-{family}.npz';z=np.load(trace);valid=z['control_sensor_valid'].astype(bool)
        xs[family]=torch.tensor(z['control_reference_beta'][valid,None],dtype=torch.float32);ys[family]=torch.tensor(z['control_requested22_rad'][valid],dtype=torch.float32)
        rows.append(dict(family=family,trace=ref(trace),valid_supported_reference_rows=int(valid.sum())))
    actor=model();initial={k:v.detach().clone() for k,v in actor.state_dict().items()}
    method=dict(schema='cove50.shallow-crouch.command-student.v1',teacher_method=ref(TEACHER_DATA/'method.json'),donors=rows,split=families,
        feature='One float32 scalar: live external requested pose blend beta in[0,1]. No measured state, time, actual joint truth, action history or reference table enters actor.',
        output='22 float32 requested output-joint target radians, exactcurrentnamedactuatororder. Runtime alone applies its unchanged clipping/delay,PD/motorloop. No learned WBC,critic orPPO.',
        joint_order=WalkingInterface().current_names,architecture='Fresh1→32→32→22 Tanh MLP, torchdefaultLinear initialization, seed138700, no PPO actor/value inheritance.',
        optimizer=dict(name='Adam',fresh_moments=True,learning_rate=.0003,betas=[.9,.999],epsilon=1e-8,updates=3000,batch_size=256,seed=138700),
        loss='Mean squared requested physical22 target error inrad², equalchannelweight. Only499validsupportedrows/family; originaltrain3/heldout4 split. No failedlearner labels.',
        selection='Last fixed3000-update checkpoint irrespectiveoftrain/heldoutloss. No automatic fittingbeyondthisbudget.',
        evaluation_families=[dict(c,seed=c['seed']+400) for c in source['frozen_learner_evaluation_families']],
        command_delivery='Evaluationharness sends beta0→1over2–4s,hold4–6,return6–8,finalhold8–10. Actoronly consumesrequestedbeta; no table/referencequery in deployedforward.',
        source=ref(Path(__file__)),limitations='Freshcommand-driven posturetrajectory student, not PPOcontinuation, feedbackpolicy,recovery,DAgger/humanexpert ororiginaldeepsquat.')
    methodpath=DATA/'method.json'
    if methodpath.exists() and json.loads(methodpath.read_text())!=method:raise ValueError('Frozen commandstudent method differs')
    write(methodpath,method)
    if args.freeze_only:print(json.dumps(dict(method=ref(methodpath))));return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'candidate.pt').exists():raise FileExistsError('Retained commandstudent candidate alreadyexists')
    start=time.monotonic();train=families['train_families'];x=torch.cat([xs[f] for f in train]);y=torch.cat([ys[f] for f in train]);opt=torch.optim.Adam(actor.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8);gen=torch.Generator().manual_seed(138700)
    def errors():
        with torch.inference_mode():return {str(f):dict(physical22_mse_rad2=float(torch.mean((actor(xs[f])-ys[f])**2)),max_abs_error_rad=float(abs(actor(xs[f])-ys[f]).max()),beta0_max_abs_error_rad=float(abs(actor(xs[f][xs[f][:,0]==0])-ys[f][xs[f][:,0]==0]).max()),beta1_max_abs_error_rad=float(abs(actor(xs[f][xs[f][:,0]==1])-ys[f][xs[f][:,0]==1]).max())) for f in xs}
    before=errors();curve=[]
    for step in range(3000):
        ids=torch.randint(len(x),(256,),generator=gen);loss=torch.mean((actor(x[ids])-y[ids])**2);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%500==0:curve.append(dict(optimizer_updates=step+1,errors=errors()))
    saved=dict(actor_state_dict=actor.state_dict(),architecture=method['architecture'],joint_order=method['joint_order'],optimizer_state_dict=opt.state_dict(),new_BC_optimizer_updates=3000,new_PPO_optimizer_updates=0,new_physics_steps=0,method=ref(methodpath),initial_actor_state_dict=initial)
    torch.save(saved,OUT/'candidate.pt');reloaded=model();reloaded.load_state_dict(torch.load(OUT/'candidate.pt',map_location='cpu',weights_only=False)['actor_state_dict'])
    with torch.inference_mode():reloaderror=float(abs(actor(x)-reloaded(x)).max())
    report=dict(method=ref(methodpath),candidate=ref(OUT/'candidate.pt'),before_errors=before,final_errors=errors(),curve=curve,optimizer_steps=sorted(set(int(s['step']) for s in opt.state.values())),
        new_BC_optimizer_updates=3000,new_PPO_optimizer_updates=0,new_simulator_transitions=0,new_physics_steps=0,new_weight_versions=1,reload_prediction_max_abs_difference_rad=reloaderror,wall_s=time.monotonic()-start)
    write(OUT/'report.json',report);print(json.dumps(dict(candidate=report['candidate'],errors=report['final_errors'],updates=report['optimizer_steps'])))


if __name__=='__main__':main()
