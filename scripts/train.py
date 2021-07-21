from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pickle as pkl
import gzip
import torch
from simple_parsing import ArgumentParser, field

import blue_zero.util as util
from blue_zero.mode import BlueEnv
from blue_zero.params import HyperParams
from blue_zero.qnet.base import QNet
from blue_zero.trainer import Trainer
from blue_zero.replay import NStepReplayMemory

torch.backends.cudnn.benchmark = True


@dataclass
class Options:
    """ options """
    # .yml file containing HyperParams
    config_file: Path = field(alias='-c', required=True)

    # .npy file containing training boards
    train_file: Path = field(alias='-t', required=True)

    # .npy file containing validation boards
    validation_file: Path = field(alias='-v', required=True)

    # where to save final trained model
    output_file: Path = field(alias='-o', required=True)

    # device to train on
    device: str = 'cpu'

    # past training results file to initialize network
    hot_start: Path = field(default=None, required=False)

    # random seed
    seed: int = field(alias='-s', default=None, required=False)


def load_envs(board_file, rotate=False, **kwargs):
    boards = np.stack(*np.load(board_file).values())

    def rot(state, k):
        return np.ascontiguousarray(np.rot90(state, k=k))

    if rotate:
        k_max = 4
    else:
        k_max = 1

    return [BlueEnv.create(board=rot(board, k), **kwargs) for board in boards
            for k in range(0, k_max)]


def main(config_file: Path, train_file: Path, validation_file: Path,
         output_file: Path, device: str = 'cpu', hot_start: Path = None,
         seed: int = None):
    if seed is not None:
        util.set_seed(seed)

    params = HyperParams.load(config_file)

    train_set = load_envs(train_file, rotate=True, **params.mode)
    validation_set = load_envs(validation_file, **params.mode)

    net = QNet.create(**params.qnet)

    if hot_start is None:
        util.init_weights(net)
    else:
        with gzip.open(hot_start, "rb") as f:
            data = pkl.load(f)
        net.load_state_dict(data["final_state"])

    memory = NStepReplayMemory(**params.replay, device=device)
    trainer = Trainer(net, train_set, validation_set, memory,
                      params.training, device=device)
    final_state, snapshots = trainer.train()

    data = dict(final_state, snapshots=snapshots)
    data['final_state'] = final_state
    data['snapshots'] = snapshots
    with gzip.open(output_file, "wb") as f:
        pkl.dump(data, f)


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()
    main(**asdict(args.options))
