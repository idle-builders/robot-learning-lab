"""Run upstream Robomimic BC-RNN on the v5 ape's checked teacher dataset."""

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np

from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.utils.torch_utils import get_torch_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Warm-start with upstream Robomimic experiment.ckpt_path")
    parser.add_argument("--append-phase", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Use upstream BC-RNN MSE instead of BC-RNN-GMM sampling")
    parser.add_argument("--feedforward", action="store_true", help="Use upstream feedforward BC without recurrent-state resets")
    parser.add_argument("--acquire", action="store_true", help="Name the open-jaw acquisition experiment")
    parser.add_argument("--place", action="store_true", help="Name the held-to-floor placement experiment")
    parser.add_argument("--normalize-obs", action="store_true",
                        help="Use Robomimic's dataset observation normalization")
    parser.add_argument("--goal-scale", type=float, default=1.0,
                        help="Scale place-task cube-to-goal coordinates in data and live replay alike")
    parser.add_argument("--goal-xy-only", action="store_true",
                        help="With --goal-scale, amplify floor target x/y while leaving height unchanged")
    parser.add_argument("--explicit-goal", action="store_true",
                        help="Use Robomimic goal_mode=last with a separate target_xy goal observation")
    args = parser.parse_args()
    if args.goal_scale != 1.0 and (not args.place or not args.append_phase or args.goal_scale <= 0):
        parser.error("--goal-scale requires --place --append-phase and a positive value")
    if args.goal_xy_only and args.goal_scale == 1.0:
        parser.error("--goal-xy-only requires --goal-scale")
    if args.explicit_goal and (not args.place or not args.append_phase or args.goal_scale != 1.0):
        parser.error("--explicit-goal requires --place --append-phase without --goal-scale")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset
    if args.append_phase:
        dataset = args.output / "teachers-with-phase.hdf5"
        with h5py.File(args.dataset, "r") as source, h5py.File(dataset, "w") as derived:
            source.copy("data", derived)
            for key, value in source.attrs.items():
                derived.attrs[key] = value
            derived.attrs["derived_from_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
            derived.attrs["additional_observation"] = "elapsed_episode_fraction"
            derived.attrs["cube_to_goal_scale"] = args.goal_scale
            derived.attrs["cube_to_goal_scale_axes"] = "xy" if args.goal_xy_only else "xyz"
            for demo in derived["data"].values():
                if args.explicit_goal:
                    goal_xy = np.asarray(demo.attrs.get("goal_xy_m", (0.08, -0.0845)), dtype=np.float32)
                    if goal_xy.shape != (2,):
                        raise ValueError(f"expected 2-D target, got {goal_xy}")
                    demo.attrs["goal_xy_m"] = goal_xy
                    for prefix in ("obs", "next_obs"):
                        demo.create_dataset(f"{prefix}/target_xy", data=np.broadcast_to(goal_xy, (len(demo[f"{prefix}/actor"]), 2)))
                for key, offset in (("obs/actor", 0), ("next_obs/actor", 1)):
                    original = demo[key][:]
                    if args.goal_scale != 1.0:
                        if original.shape[1] != 72:
                            raise ValueError(f"expected 72-D place actor before scaling, got {original.shape}")
                        original[:, 47:49 if args.goal_xy_only else 50] *= args.goal_scale
                    phase = ((np.arange(len(original), dtype=np.float32) + offset) / len(original))[:, None]
                    del demo[key]
                    demo.create_dataset(key, data=np.concatenate((original, phase), axis=1))
    cfg = config_factory("bc")
    lesson = "place" if args.place else "acquire" if args.acquire else "pregrasp"
    cfg.experiment.name = "ape_jaw_v5_" + lesson + "_bc" + ("_ff" if args.feedforward else "_rnn") + ("_phase" if args.append_phase else "") + ("_mse" if args.deterministic else "") + ("_norm" if args.normalize_obs else "") + (f"_goal{args.goal_scale:g}{'xy' if args.goal_xy_only else ''}" if args.goal_scale != 1.0 else "") + ("_explicitgoal" if args.explicit_goal else "") + ("_warmstart" if args.init_checkpoint else "")
    if args.init_checkpoint:
        if not args.init_checkpoint.is_file():
            parser.error(f"missing --init-checkpoint: {args.init_checkpoint}")
        cfg.experiment.ckpt_path = str(args.init_checkpoint)
    cfg.experiment.validate = False
    cfg.experiment.rollout.enabled = False  # fresh MJLab replay is a separate process
    cfg.experiment.logging.terminal_output_to_txt = False
    cfg.experiment.logging.log_tb = False
    cfg.experiment.save.every_n_epochs = args.save_every
    cfg.experiment.save.on_best_validation = False
    cfg.experiment.save.on_best_rollout_success_rate = False
    cfg.experiment.epoch_every_n_steps = 50
    cfg.train.data = str(dataset)
    cfg.train.output_dir = str(args.output)
    cfg.train.num_epochs = args.epochs
    cfg.train.batch_size = 64
    cfg.train.seq_length = 1 if args.feedforward else 10
    cfg.train.hdf5_filter_key = None
    cfg.train.hdf5_validation_filter_key = None
    cfg.train.hdf5_load_next_obs = args.explicit_goal
    cfg.train.hdf5_normalize_obs = args.normalize_obs
    cfg.train.goal_mode = "last" if args.explicit_goal else None
    cfg.train.num_data_workers = 0
    cfg.observation.modalities.obs.low_dim = ["actor"]
    cfg.observation.modalities.obs.rgb = []
    cfg.observation.modalities.goal.low_dim = ["target_xy"] if args.explicit_goal else []
    cfg.observation.modalities.goal.rgb = []
    if not args.feedforward:
        cfg.algo.actor_layer_dims = ()
    cfg.algo.gmm.enabled = not args.deterministic
    cfg.algo.gmm.num_modes = 5
    cfg.algo.rnn.enabled = not args.feedforward
    cfg.algo.rnn.horizon = 10
    cfg.algo.rnn.hidden_dim = 400
    cfg.algo.rnn.rnn_type = "LSTM"
    cfg.algo.rnn.num_layers = 2
    (args.output / "config.json").write_text(cfg.dump() + "\n")
    train(cfg, device=get_torch_device(try_to_use_cuda=True))


if __name__ == "__main__":
    main()
