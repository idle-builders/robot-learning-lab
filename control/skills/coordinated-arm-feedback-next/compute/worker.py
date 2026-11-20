"""Same exercised stage/join/pack/rescue worker; separate next-lesson names."""
import importlib.util
from pathlib import Path


def source():
    path = Path(__file__).resolve().parents[4] / 'control/skills/coordinated-arm-feedback/compute/worker.py'
    spec = importlib.util.spec_from_file_location('feedback_r1_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = module.source()
    changes = [('run_coordinated_arm_feedback', 'run_coordinated_arm_feedback_next', 2),
               ('coordinated-arm-feedback', 'coordinated-arm-feedback-next', 7),
               ('Training uses one shared4row RNG stream; not4independently reseeded streams.',
                'Training has eight slots sharing one Runtime RNG stream and four repeated physical entry templates; hidden scenario RNGs are separate, not independent mechanical families.', 1)]
    for old, new, count in changes:
        if raw.count(old) != count:
            raise ValueError('Exact retained worker anchor differs: ' + old)
        raw = raw.replace(old, new)
    return raw


if __name__ == '__main__':
    exec(compile(source(), str(__file__), 'exec'), globals())
