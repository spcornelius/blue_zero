from enum import IntEnum

__all__ = ['Status', 'color', 'pad_frac', 'screen_size']


class Status(IntEnum):
    wall = 0
    alive = 1
    dead = 2
    attacked = 3


black = (0, 0, 0, 255)
orange = (255, 80, 0, 255)
blue = (20, 165, 195, 255)
green = (160, 185, 115, 255)

# fraction of the smaller of (screen height, screen width) that should
# be used for padding around each cell
pad_frac = 0.075

screen_size = (1200, 1200)

color = {Status.wall: black,
         Status.attacked: orange,
         Status.dead: blue,
         Status.alive: green
         }
