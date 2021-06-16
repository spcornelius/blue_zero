from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from simple_parsing import ArgumentParser, field

import blue_zero.util as util
from blue_zero.env import BlueEnv
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

    # random seed
    seed: int = field(alias='-s', default=None, required=False)


def load_envs(board_file, **kwargs):
    boards = np.stack(*np.load(board_file).values())
    return [BlueEnv.create(board=board, **kwargs) for board in boards]


def main(config_file: Path, train_file: Path, validation_file: Path,
         output_file: Path, device: str = 'cpu', seed: int = None):
    if seed is not None:
        util.set_seed(seed)

    params = HyperParams.load(config_file)

    train_set = load_envs(train_file, **params.mode)
    validation_set = load_envs(validation_file, **params.mode)

    net = QNet.create(**params.qnet)
    util.init_weights(net)
    memory = NStepReplayMemory(**params.replay, device=device)
    trainer = Trainer(net, train_set, validation_set, memory,
                      params.training, device=device)
    trained_net = trainer.train()
    trained_net.save(output_file)


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()
    main(**asdict(args.options))
