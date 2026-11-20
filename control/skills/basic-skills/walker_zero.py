"""Two bounded zero-command stance visits by the current adapted R3 walker.

Keeps the checkpoint's exact81/24 raw-history/target contract; no posture
normalization, smoothing or action authority is substituted.
"""
from pathlib import Path
import argparse,json
import torch
import numpy as np
from adapter import ROOT,WalkingInterface,ref
from run import episode,write,HERE,DATA,OUT

CHECKPOINT=ROOT/'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/evidence/fit-through-200/candidate.pt'


class Walker:
    action_width=24
    def __init__(self):
        self.mapping=WalkingInterface();self.offset=self.mapping.home.numpy().copy()
        self.saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        if ref(CHECKPOINT)['sha256']!='467b12a17a71f06d951294b3d9185503ea78c6edda431ae7e7870aa648fc02bd':raise ValueError('Candidate identity changed')
        self.state=self.saved['actor_state_dict'];self.reset()

    def reset(self):self.previous_raw=np.zeros(24,dtype=np.float32)

    def observe(self,packet):
        if packet is None:return np.zeros(81,dtype=np.float32)
        v=packet['values']
        return self.mapping.observe(torch.tensor(v[:22][None],dtype=torch.float32),torch.tensor(v[22:44][None],dtype=torch.float32),
            torch.tensor(v[44:47][None],dtype=torch.float32),torch.tensor(v[47:50][None],dtype=torch.float32),
            torch.tensor(self.previous_raw[None]),torch.zeros(1,3))[0].numpy()

    def action(self,obs):
        s=self.state;x=(torch.tensor(obs[None])-s['obs_normalizer._mean'])/(s['obs_normalizer._std']+.01)
        with torch.inference_mode():
            for layer in (0,2,4,6):
                x=torch.nn.functional.linear(x,s[f'mlp.{layer}.weight'],s[f'mlp.{layer}.bias'])
                if layer!=6:x=torch.nn.functional.elu(x)
            target=self.mapping.target(x)[0].numpy()
        raw=x[0].numpy();self.previous_raw=raw.copy();oldtarget=self.offset.copy();oldtarget[:10]+=.3*raw[:10]
        for name,limits in self.mapping.source_clips.items():
            if name in self.mapping.source_names:
                i=self.mapping.source_names.index(name);oldtarget[i]=np.clip(oldtarget[i],*limits)
        return raw,raw.copy(),oldtarget,target


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['probe','fresh'],default='probe');args=p.parse_args()
    torch.set_num_threads(1);cfg=json.loads((HERE/'stand.json').read_text());cfg['controllers']=['walker-zero']
    if args.phase=='probe':cfg['starts']=cfg['starts'][:2]
    else:
        for case in cfg['starts']:case['seed']+=300
    walker=Walker();original=json.loads((DATA/'method.json').read_text())
    label='walker-zero' if args.phase=='probe' else 'walker-zero-fresh'
    out=OUT/label;data=DATA/label;out.mkdir(exist_ok=True);data.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Native visits already retained')
    method=dict(schema='cove50.basic-standing.walker-zero.v1',selected=original['selected'],pointer=original['pointer'],configuration=cfg,
        checkpoint=ref(CHECKPOINT),actor_contract='Exact adapted R3 actor81/raw24 with actualraw21+3 history and source Interface.target, fixed commandedvelocity(0,0,0); first20ms bootstrap retained.',
        inherited_training=walker.saved['infos'],new_optimizer_updates=0,
        initial_state_pairing='Same CAD-neutral family137001/137002, exact same Runtime seed/reset as original held neutral comparator; those originals not rerun.' if args.phase=='probe' else 'Four fresh reserved families137301..137304; same predefined variation magnitudes, no new neutral visits. Earlier neutral observations provide reference bounds, not a same-state causal pair.',
        sources=[ref(Path(__file__)),ref(HERE/'run.py'),ref(HERE/'adapter.py'),ref(ROOT/'control/skills/walker/policy/adapter.py')],
        limitations='This is a zero-velocity probe of a current-body walking actor, not a retrained posture actor or standup/recovery demonstration.')
    path=data/'method.json';write(path,method)
    rows=[episode(cfg,case,'walker-zero',walker,out,method) for case in cfg['starts']]
    write(out/'report.json',dict(method=ref(path),episodes=rows,simulator_transitions=sum(r['controls'] for r in rows),
        native_physics_steps=sum(r['native_physics_steps'] for r in rows),optimizer_updates=0,independent_start_families=len(cfg['starts']),world_slots=1,
        new_independent_start_families=0 if args.phase=='probe' else 4,
        schema_note='Legacy recorder keys control_observation72/raw_action21/filter21 carry actualwidth81/24/24 for this source profile; no tensor truncation. source_target21 remains source leg target; actual physical target22 retained.'))


if __name__=='__main__':main()
