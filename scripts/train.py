from dataclasses import dataclass

import numpy as np
from path import Path
from simple_parsing import ArgumentParser, field
from blue_zero.hyper import HyperParams
from blue_zero.env import Blue
from blue_zero.net.dqn import DQN
from blue_zero.trainer import Trainer


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


def load_envs(board_file):
    return list(map(Blue, np.load(board_file)))


def main(config_file: Path, train_file: Path, validation_file: Path,
         output_file: Path, device: str = 'cpu'):
    train_set = load_envs(train_file)
    validation_set = load_envs(validation_file)
    hp = HyperParams.load(config_file)
    net = DQN(**vars(hp.net_params))
    trainer = Trainer(net, train_set, validation_set, hp.train_params,
                      device=device)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()
    main(**vars(args.options))
