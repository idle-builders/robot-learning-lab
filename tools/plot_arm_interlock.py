"""Present retained native waveforms without rerunning or embellishing the model."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(source,output):
    report=json.loads((source/'report.json').read_text())
    case=next(c for c in report['cases'] if c['name']=='held_through_stop' and c['debounce_s']==.08 and c['recovery_s']==.18)
    if report['status']!='pass' or case['status']!='pass':raise ValueError('Trace is not a passing native case')
    data=np.loadtxt(source/case['trace'],skiprows=1)
    fig,ax=plt.subplots(6,1,figsize=(11,6.5),sharex=True,layout='constrained')
    for axis,column,label,color in zip(ax,[7,6,3,1,8,2],['Arm button (low = pressed)','Stop contact (high = closed)','Recovery complete','Latched arm authority','Software power request','Motor enable (logic only)'],['#588ac4','#dc6e55','#9c85c0','#69ae91','#588ac4','#d4a24a']):
        axis.plot(data[:,0],data[:,column],color=color,lw=2)
        axis.set_ylim(-.25,3.65);axis.set_yticks([0,3.3]);axis.set_ylabel('V',rotation=0)
        axis.text(.01,.84,label,transform=axis.transAxes,fontsize=9)
        axis.spines[['top','right']].set_visible(False);axis.grid(alpha=.15)
        axis.axvspan(.85,.9,color='#dc6e55',alpha=.12)
    ax[-1].set_xlabel('Time in seconds · synthetic inputs, generic ngspice logic models')
    ax[0].set_title('Holding arm through a stop does not restart power',loc='left',fontweight='bold',pad=18)
    ax[-1].annotate('Release',xy=(1.5,0),xytext=(1.36,1.5),arrowprops={'arrowstyle':'->'},fontsize=9)
    ax[-1].annotate('New press qualifies',xy=(1.78,3.3),xytext=(1.57,1.4),arrowprops={'arrowstyle':'->'},fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    fig.savefig(output,dpi=160)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
