import gzip
import pickle as pkl
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pygame
import torch
from simple_parsing import ArgumentParser, field

from blue_zero.agent import QAgent
from blue_zero.config import Status
from blue_zero.mode import mode_registry
from blue_zero.qnet import QNet
from blue_zero.gui import BlueGUI


@dataclass
class Options:
    # .pkl.gz file containing training results
    file: Path = field(alias='-f', required=True)

    # size of board to play on
    n: int = field(alias='-n', required=True)

    # green probability
    p: float = field(alias='-p', required=True)

    # game mode
    mode: int = field(required=True)


def main(file: Path, n: int, p: float, mode: int,
        **kwargs):
    with gzip.open(file, "rb") as f:
        data = pkl.load(f)
    net = QNet.create(**data['qnet_params'])
    net.load_state_dict(data['final_state'])
    agent = QAgent(net)
    gui = BlueGUI((n, n))
    env = mode_registry[mode].from_random((n, n), p, gui=gui, **kwargs)

    print("Click anywhere to start playing.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and not env.done:
                a = agent.get_action(env.state)
                if a is not None:
                    i, j = a
                    if env.state[Status.alive, i, j]:
                        env.step(a)


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()
    kwargs = asdict(args.options)
    main(**kwargs)
