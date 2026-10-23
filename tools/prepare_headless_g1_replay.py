"""Keep upstream replay; replace only its headless keyboard and Arena progress hooks."""

import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
replacements = {
    "teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))":
        'teleop_interface = type("HeadlessKeyboard", (), {"add_callback": lambda *args: None, "reset": lambda *args: None})()',
    "env = gym.make(args_cli.task, cfg=env_cfg).unwrapped":
        "env = gym.make(args_cli.task, cfg=env_cfg).unwrapped\n"
        "    from isaaclab.managers import ManagerTermBase\n"
        "    if success_term is not None and isinstance(success_term.func, type) and issubclass(success_term.func, ManagerTermBase):\n"
        "        success_term.func = success_term.func(success_term, env)",
    "env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)":
        "env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)\n"
        "                            if success_term is not None and hasattr(success_term.func, 'reset'):\n"
        "                                success_term.func.reset(torch.tensor([env_id], device=env.device))",
    "                env.step(actions)":
        "                env.step(actions)\n"
        "                if success_term is not None:\n"
        "                    success_term.func(env, **success_term.params)",
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise SystemExit(f"Expected one upstream insertion point, found {source.count(old)}: {old}")
    source = source.replace(old, new)
Path(sys.argv[2]).write_text(source)
