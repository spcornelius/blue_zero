from dataclasses import dataclass, asdict

import numpy as np
import torch
from path import Path
from simple_parsing import ArgumentParser, field

import blue_zero.util as util
from blue_zero.env.util import env_cls, ModeOptions
from blue_zero.params import HyperParams
from blue_zero.qnet.base import QNet
from blue_zero.trainer import Trainer

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

    # game mode
    mode: int = field(required=True)

    # device to train on
    device: str = 'cpu'

    # random seed
    seed: int = field(alias='-s', default=None, required=False)


def load_envs(board_file, mode, **kwargs):
    return list(map(lambda board: env_cls[mode](board, **kwargs),
                    np.load(board_file)))


def main(config_file: Path, train_file: Path, validation_file: Path,
         output_file: Path, mode: int, device: str = 'cpu', seed: int = None,
         **kwargs):
    if seed is not None:
        util.set_seed(seed)
    train_set = load_envs(train_file, mode, **kwargs)
    validation_set = load_envs(validation_file, mode, **kwargs)
    params = HyperParams.load(config_file)
    qnet_type = params.qnet.pop('type')
    net = QNet.create(qnet_type, **params.qnet)
    trainer = Trainer(net, train_set, validation_set, params.training,
                      device=device)
    trained_net = trainer.train()
    trained_net.save(output_file)


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    parser.add_arguments(ModeOptions, "mode_options")
    args = parser.parse_args()
    kwargs = asdict(args.options)
    kwargs.update(args.mode_options.get_kwargs(args.options.mode))
    main(**kwargs)

