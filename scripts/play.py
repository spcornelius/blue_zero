import gzip
import pickle as pkl
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from time import sleep

import pygame
from simple_parsing import ArgumentParser, field

from blue_zero.agent import QAgent
from blue_zero.config import Status
from blue_zero.gui import BlueGUI
from blue_zero.mode import mode_registry
from blue_zero.qnet import QNet


@dataclass
class Options:
    # size of board to play on
    n: int = field(alias='-n', required=True)

    # green probability
    p: float = field(alias='-p', required=True)

    # game mode
    mode: int = field(required=True)

    # .pkl.gz file containing training results (for computer mode)
    file: Path = field(alias='-f', required=False)


def main(n: int, p: float, mode: int, file: Path = None,
         **kwargs):
    machine_mode = file is not None
    if machine_mode:
        net = QNet.load(file)
        agent = QAgent(net)

    gui = BlueGUI((n, n))

    while True:
        env = mode_registry[mode].from_random((n, n), p, gui=gui, **kwargs)
        print("Click anywhere to start playing.")

        game_over = False
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # has the effect of requiring an extra button click
                    # after game over before erasing the screen
                    if env.done:
                        game_over = True
                    else:
                        if machine_mode:
                            a = agent.get_action(env.state)
                        else:
                            a = gui.get_clicked_square(pygame.mouse.get_pos())
                        if a is not None:
                            i, j = a
                            if env.state[Status.alive, i, j]:
                                env.step(a)

        # prevents a multiple button click from activating one of the
        # two buttons that will appear in the center
        sleep(0.2)

        pygame.event.clear()
        gui.draw_game_over()
        buttons = gui.buttons
        play_again, quit = buttons.buttons
        quit.clicked = False
        play_again.clicked = False

        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()

            buttons.listen(events)
            buttons.draw()
            pygame.display.update()
            if play_again.clicked or quit.clicked:
                break

        if quit.clicked:
            break


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()
    kwargs = asdict(args.options)
    main(**kwargs)
