"""Keep upstream Mimic annotation; inject inspected manual marks without a GUI."""

import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
replacements = {
    "env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped":
        "env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped\n"
        "    success_term = None  # Native recording and separate Arena replay already validated success.",
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise SystemExit(f"Expected one upstream insertion point, found {source.count(old)}: {old}")
    source = source.replace(old, new)

# The pinned G1 Mimic environment has no automatic subtask detector. Replace only
# manual keyboard marking with inspected frame indices; the upstream replay,
# recorder, HDF5 export, and Mimic semantics stay in use.
start = source.index("def annotate_episode_in_manual_mode(")
end = source.index("\n\nif __name__ ==", start)
source = source[:start] + '''def annotate_episode_in_manual_mode(
    env, episode, success_term=None, subtask_term_signal_names={}, subtask_start_signal_names={}
):
    import json
    import os

    if subtask_start_signal_names and any(subtask_start_signal_names.values()):
        raise ValueError("Only termination signals are supported by these inspected marks")
    marks = json.load(open(os.environ["ARENA_ANNOTATION_MARKS"]))
    expected = {name for names in subtask_term_signal_names.values() for name in names}
    if set(marks) != expected:
        raise ValueError(f"Expected marks {sorted(expected)}, got {sorted(marks)}")
    if not replay_episode(env, episode, success_term):
        return False
    annotated = env.recorder_manager.get_episode(0)
    count = len(episode.data["actions"])
    for name, index in marks.items():
        if not isinstance(index, int) or not 0 < index < count:
            raise ValueError(f"Invalid mark {name}={index} for {count} actions")
        flags = torch.ones(count, dtype=torch.bool)
        flags[:index] = False
        annotated.add(f"obs/datagen_info/subtask_term_signals/{name}", flags)
    print(f"Applied inspected manual subtask marks: {marks}")
    return True
''' + source[end:]
Path(sys.argv[2]).write_text(source)
