import sys
from dataclasses import dataclass, asdict
from time import sleep

import pygame
from path import Path
from simple_parsing import ArgumentParser, field

from blue_zero.agent import Agent
from blue_zero.env import ModeOptions
from blue_zero.qnet.base import QNet
from blue_zero.env.util import env_cls


@dataclass
class Options:
    # .pt file containing a trained model
    file: Path = field(alias='-f', required=True)

    # size of board to play on
    n: int = field(alias='-n', required=True)

    # green probability
    p: float = field(alias='-p', required=True)

    # game mode
    mode: int = field(required=True)

    # time delay between taking moves
    pause: float = 0.2


def main(file: Path, n: int, p: float, mode: int,
         pause: float = 0.2, **kwargs):
    net = QNet.load(file)
    agent = Agent(net)
    env = env_cls[mode].from_random((n, n), p, with_gui=True, **kwargs)

    started = False
    print("Click anywhere to start playing.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and not started:
                print("Playing!")
                started = True
        if not started:
            continue

        if not env.done:
            a = agent.get_action(env.state, eps=0.0)
            env.step(a)
            sleep(pause)


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    parser.add_arguments(ModeOptions, "mode_options")
    args = parser.parse_args()
    kwargs = asdict(args.options)
    kwargs.update(args.mode_options.get_kwargs(args.options.mode))
    main(**kwargs)
