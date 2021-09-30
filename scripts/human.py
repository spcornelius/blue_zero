import sys
from dataclasses import dataclass, asdict
from time import sleep

import pygame
from path import Path
from simple_parsing import ArgumentParser, field

from blue_zero.agent import Agent
from blue_zero.env import ModeOptions
from blue_zero.net.dqn import DQN
from blue_zero.env.util import env_cls


@dataclass
class Options:

    # size of board to play on
    n: int = field(alias='-n', required=True)

    # green probability
    p: float = field(alias='-p', required=True)

    # game mode
    mode: int = field(required=True)



def main(n: int, p: float, mode: int,
          **kwargs):
    env = env_cls[mode].from_random((n, n), p, with_gui=True, **kwargs)

    started = False
    print("Click anywhere to start playing.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or env.done:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and not started:
                print("Playing!")
                a = env.gui.get_clicked_square(pygame.mouse.get_pos())
                if a is not None:
                    env.step(a)
                
        if env.done:
            print("Click anywhere to exit")
            
           
           


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    parser.add_arguments(ModeOptions, "mode_options")
    args = parser.parse_args()
    kwargs = asdict(args.options)
    kwargs.update(args.mode_options.get_kwargs(args.options.mode))
    main(**kwargs)
