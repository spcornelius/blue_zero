import sys
from dataclasses import dataclass, asdict

import pygame
from simple_parsing import ArgumentParser, field
from blue_zero.env import mode_registry
from blue_zero.config import Status


@dataclass
class Options:
    # size of board to play on
    n: int = field(alias='-n', required=True)

    # green probability
    p: float = field(alias='-p', required=True)

    # game mode
    mode: int = field(required=True)


def main(n: int, p: float, mode: int, **kwargs):
    env = mode_registry[mode].from_random((n, n), p, show_gui=True, **kwargs)
    gui = env.gui

    print("Click anywhere to start playing.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and not env.done:
                a = gui.get_clicked_square(pygame.mouse.get_pos())
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
