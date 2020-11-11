import sys
from dataclasses import dataclass
from time import sleep

import pygame
from path import Path
from simple_parsing import ArgumentParser, field

from blue_zero.agent import Agent
from blue_zero.env.blue import Blue
from blue_zero.net.dqn import DQN


@dataclass
class Options:
    # .pt file containing a trained model
    file: Path = field(alias='-f', required=True)

    # size of board to play on
    n: int = field(alias='-n', required=True)

    # green probability
    p: float = field(alias='-p', required=True)

    # time delay between taking moves
    pause: float = 0.2


def main(file: Path, n: int, p: float,
         pause: float = 0.2):
    net = DQN.load(file)
    agent = Agent(net)
    env = Blue.from_random((n, n), p, with_gui=True)

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
    parser = ArgumentParser()
    parser.add_arguments(Options, "options")
    options = parser.parse_args().options
    main(options.file, options.n, options.p,
         pause=options.pause)
