"""Register our downstream task, then invoke the unmodified upstream trainer."""
import os
from pathlib import Path
import runpy
import sys
import gymnasium as gym

os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
os.environ['OMNI_KIT_ALLOW_ROOT'] = '1'
operation = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1] in {'train', 'eval', 'export'} else 'train'
gym.register(id='RobotLab-Pip-Agile-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True, kwargs={'env_cfg_entry_point': 'pip_cfg:PipEnvCfg',
                                     'rsl_rl_cfg_entry_point': 'pip_cfg:PipRunnerCfg'})
scripts = Path(os.environ['AGILE_CHECKOUT']) / 'scripts'
sys.path.insert(0, str(scripts))
runpy.run_path(str(scripts / {'train': 'train.py', 'eval': 'eval.py', 'export': 'export_policy_leapp.py'}[operation]), run_name='__main__')
