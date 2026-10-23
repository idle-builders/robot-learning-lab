"""Train upstream Robomimic feedforward BC from accepted ape foot demos."""

import argparse
from pathlib import Path

from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.utils.torch_utils import get_torch_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = config_factory("bc")
    cfg.experiment.name = "ape_foot_transition_bc_ff"
    cfg.experiment.validate = False
    cfg.experiment.rollout.enabled = False
    cfg.experiment.logging.terminal_output_to_txt = False
    cfg.experiment.logging.log_tb = False
    cfg.experiment.save.every_n_epochs = 10
    cfg.experiment.save.on_best_validation = False
    cfg.experiment.save.on_best_rollout_success_rate = False
    cfg.experiment.epoch_every_n_steps = 100
    cfg.train.data = str(args.dataset)
    cfg.train.output_dir = str(args.output)
    cfg.train.num_epochs = args.epochs
    cfg.train.batch_size = 64
    cfg.train.seq_length = 1
    cfg.train.hdf5_filter_key = None
    cfg.train.hdf5_validation_filter_key = None
    cfg.train.hdf5_load_next_obs = False
    cfg.train.hdf5_normalize_obs = True
    cfg.train.num_data_workers = 0
    cfg.observation.modalities.obs.low_dim = ["actor"]
    cfg.observation.modalities.obs.rgb = []
    cfg.observation.modalities.goal.low_dim = []
    cfg.observation.modalities.goal.rgb = []
    cfg.algo.gmm.enabled = False
    cfg.algo.rnn.enabled = False
    (args.output / "config.json").write_text(cfg.dump() + "\n")
    train(cfg, device=get_torch_device(try_to_use_cuda=True))


if __name__ == "__main__":
    main()
