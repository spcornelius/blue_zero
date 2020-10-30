from typing import Tuple

import numpy as np
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
from torch import Tensor

import blue_zero.config as cfg


class BlueGUI(object):

    def __init__(self, board_size: Tuple[int, int],
                 screen_size: Tuple[int, int] = cfg.screen_size):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(screen_size)
        screen_h, screen_w = screen_size
        self.screen_size = (screen_h, screen_w)

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the maze
        self.board_layer = pygame.Surface(
            self.screen.get_size()).convert_alpha()

        self.board_layer.fill((0, 0, 0, 0,))
        self.board_h, self.board_w = board_size
        self.cell_w = (1 - cfg.pad_frac) * screen_w / self.board_w
        self.cell_h = (1 - cfg.pad_frac) * screen_h / self.board_h
        self.pad_w = cfg.pad_frac * screen_w / (self.board_w + 1)
        self.pad_h = cfg.pad_frac * screen_h / (self.board_h + 1)
        self.rects = dict()

    def draw_board(self, state: Tensor) -> None:
        board_h, board_w = state.shape
        assert board_h == self.board_h
        assert board_w == self.board_w

        self.rects = dict()
        for status in cfg.Status:
            for ij in np.argwhere(state == status):
                i, j = tuple(ij)
                left = self.cell_w * j + self.pad_w * (j + 1)
                top = self.cell_h * i + self.pad_h * (i + 1)
                width, height = self.cell_w, self.cell_h
                r = pygame.Rect(left, top, width, height)
                self.rects[i, j] = r
                pygame.draw.rect(self.board_layer, cfg.color[status], r, 0)

        # update the screen
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.board_layer, (0, 0))

        pygame.display.flip()

    def get_clicked_square(self, pos):
        for ij, r in self.rects.items():
            if r.collidepoint(pos):
                return ij
