"""One fresh recurrent command-memory fit; synthetic rows are not simulation."""
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from adapter import ROOT,ref
from run import HERE,write
from arm_command import CommandNet,canonical,labels_for
from arms_trial import CASES
from arms_donors import DATA as DONOR_DATA

DATA=ROOT/'data/basic-skills-retry-20261027/arms/command-gru2000'
OUT=ROOT/'artifacts/cove50-basic-skills-retry/arms-20261027/command-gru2000'


def schedules(seed,n,canonical_first,forbidden=None):
    rng=np.random.default_rng(seed);result=[canonical(c) for c in CASES[:3]] if canonical_first else [];seen=set(forbidden or [])
    seen.update(labels_for(s)[0].tobytes() for s in result)
    while len(result)<n:
        sides=[]
        for _ in range(2):
            a=int(rng.integers(25,175));b=min(465,a+int(rng.integers(15,200)));spans=[[a,b]]
            if b<350 and rng.random()<.5:
                c=int(rng.integers(b+10,430));d=min(499,c+int(rng.integers(15,100)));spans.append([c,d])
            if rng.random()<.15:spans=[]
            sides.append(spans)
        candidate=dict(name='synthetic-'+str(seed)+'-'+str(len(result)),intervals_L_R=sides);key=labels_for(candidate)[0].tobytes()
        if key in seen:continue
        seen.add(key);result.append(candidate)
    return result


def generate():
    train=schedules(140600,96,True);held=schedules(140601,24,False,{labels_for(s)[0].tobytes() for s in train})
    schedules_path=DATA/'schedules.json';payload=dict(train=train,heldout=held)
    if schedules_path.exists() and json.loads(schedules_path.read_text())!=payload:raise ValueError('Frozen schedules differ')
    write(schedules_path,payload)
    all_schedules=train+held;flags=[];alpha=[]
    for s in all_schedules:
        x,y=labels_for(s);flags.append(x);alpha.append(y)
    x=np.stack(flags);y=np.stack(alpha);target=DATA/'generated-labels.npz';np.savez_compressed(target,flags=x,alpha=y)
    return x,y,ref(schedules_path),ref(target)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');p.add_argument('--generate-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    x,y,schedule_ref,label_ref=generate()
    if args.generate_only:print(json.dumps(dict(schedules=schedule_ref,generated=label_ref)));return
    donor=json.loads((DONOR_DATA/'method.json').read_text());evaluation=copy.deepcopy(donor['evaluation_families'])
    for case in evaluation:case['seed']+=400
    shifted=[dict(name='bilateral-shifted',case='bilateral-both',intervals_L_R=[[[75,275]],[[75,275]]]),dict(name='left-shifted',case=CASES[1],intervals_L_R=[[[125,325]],[]]),dict(name='right-shifted',case=CASES[2],intervals_L_R=[[],[[50,250]]]),dict(name='asynchronous',case='independent-both',intervals_L_R=[[[75,250]],[[150,325]]])]
    contract=dict(schema='cove50.basic-arms.fresh-command-gru2000.v1',schedules=schedule_ref,generated_labels=label_ref,
        architecture='Fresh torchGRU(input2,hidden16,batch_first),linear16→2,clamp[0,1]. No normalizer,critic,oldactor orPPOoptimizer. Headweight uniform±.01,bias.05; remainingPyTorchdefault initialization,seed140600.',
        inference='Only current2binaryexternalflags andhiddenstate; resethiddenzeros; noelapsedtime/scheduledalpha/roottruth/proprioception/table atactorinference. Fixed50Hz invocation; originalsourceEMA.2bilateral/.05independent, raw21 shoulder13/18=-1.2predictedalpha,other19zero, sourceclip/sign andcurrentneutral20axes unchanged.',
        timing='500 scheduledframes0..9.98s; frame0 invalid20msbootstrap holdsactualq andNO recurrentcall. Train/infer499calls. Teacheralpha[k]=alpha[k−1]+clip(flags[k−1]−alpha[k−1],±.02/1.5),initial0; canonical2salpha0,6salpha1.',
        optimizer=dict(algorithm='fresh Adam',learning_rate=.001,betas=[.9,.999],eps=1e-8,updates=2000,batch_sequences=16,seed=140600,loss='MSE predicted2alpha onall499validrows, wholecausalsequences andBPTT; nohiddenreusebetweenbatchsequences.'),
        split=dict(train_schedules=96,heldout_schedules=24,scheduled_frames_per_sequence=500,valid_recurrent_rows_per_sequence=499,heldout_sampling=0,unique_flag_trajectories_train=96,unique_flag_trajectories_heldout=24,train_heldout_full_flag_byte_overlap=0),
        selection='Fixed final2000update checkpoint regardless ofloss, noautomaticextrafit. Heldoutdata reportonly.',evaluation_families=evaluation,shifted_nominal_schedules=shifted,shifted_nominal_seed=140705,
        sources=[ref(HERE/n) for n in ('arm_command.py','fit_arm_command.py','evaluate_arm_command.py','arms_profile.py','run.py','adapter.py')],
        source_motion=ref(ROOT/'data/basic-skills-retry-20261027/arms/model-reference/method.json'),
        limits='Freshlearned recurrentintention/trajectorycomponent overjointPD, notretained72/75feedbackactor,valuelearning,WBC,recovery,humanDAgger/correctiveexpert. Synthetic modelgeneratedlabels are notsimulator transitions.',
        cleanup='After fit verify --generate-only reproduces the exact generated-label hash, retain schedules/config/source/compact regeneration receipt, then deleteunusedgenerated-labels.npz; never coldarchive/upload this cheap derived array.')
    path=DATA/'method.json'
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Frozen recurrent method differs')
    write(path,contract)
    if args.freeze_only:print(json.dumps(dict(method=ref(path))));return
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise FileExistsError('Original commandfit outputs retained')
    torch.manual_seed(140600);model=CommandNet();initial=copy.deepcopy(model.state_dict());optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8);generator=torch.Generator().manual_seed(140600)
    tx=torch.tensor(x[:,1:]);ty=torch.tensor(y[:,1:]);start=time.monotonic();curve=[]
    def errors():
        with torch.inference_mode():
            pred,_=model(tx);error=pred-ty
            return {name:dict(mse=float(error[a:b].square().mean()),MAE=float(error[a:b].abs().mean()),max_abs=float(error[a:b].abs().max())) for name,a,b in [('train',0,96),('heldout',96,120)]}
    initial_errors=errors()
    for step in range(2000):
        ids=torch.randint(96,(16,),generator=generator);prediction,_=model(tx[ids]);loss=(prediction-ty[ids]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite recurrent loss')
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%200==0:
            point=dict(optimizer_updates=step+1,errors=errors(),wall_s=time.monotonic()-start);curve.append(point);print(json.dumps(point),flush=True)
    final_errors=errors();checkpoint=OUT/'candidate.pt';torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),method=ref(path),new_optimizer_updates=2000,initial_state_dict=initial),checkpoint)
    reloaded=CommandNet();reloaded.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'])
    with torch.inference_mode():
        prediction,_=model(tx);replay,_=reloaded(tx);parity=float(abs(prediction-replay).max());one=tx[:1];h=None;pieces=[]
        for k in range(one.shape[1]):v,h=reloaded(one[:,k:k+1],h);pieces.append(v)
        streamerr=float(abs(torch.cat(pieces,1)-replay[:1]).max())
    write(OUT/'report.json',dict(method=ref(path),candidate=ref(checkpoint),initial_errors=initial_errors,final_errors=final_errors,curve=curve,
        optimizer_updates=2000,optimizer_steps=sorted(set(int(v['step']) for v in optimizer.state.values())),new_trained_weight_versions=1,initial_untrained_parameter_sets=1,
        sampled_sequences=32000,sampled_scheduled_synthetic_rows=16000000,sampled_valid_synthetic_label_rows=15968000,distinct_train_scheduled_rows=48000,distinct_train_valid_rows=47904,heldout_scheduled_rows=12000,heldout_valid_rows=11976,
        new_simulator_transitions=0,new_physics_steps=0,PPO_updates=0,reload_output_max_abs_difference=parity,stream_vs_sequence_max_abs_difference=streamerr,wall_s=time.monotonic()-start))
    print(json.dumps(dict(candidate=ref(checkpoint),final_errors=final_errors)),flush=True)

if __name__=='__main__':main()
