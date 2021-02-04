from blue_zero.env.mode0 import BlueMode0
from blue_zero.env.mode3 import BlueMode3
from blue_zero.env.mode4 import BlueMode4
from dataclasses import dataclass, asdict
from simple_parsing import choice


__all__ = []
__all__.extend(['env_cls', 'ModeOptions', 'Mode3Options', 'Mode4Options'])


env_cls = {
    0: BlueMode0,
    3: BlueMode3,
    4: BlueMode4,
}


@dataclass
class Mode3Options:
    """ options for mode 3 """
    # direction of current
    direction: str = choice('horizontal', 'vertical', 'both',
                            default='horizontal')


@dataclass
class Mode4Options:
    """ options for mode 4 """
    # noodle dimension
    K: float = 2.0

@dataclass
class ModeOptions:
    """" options for specific BLUE environments """
    mode3: Mode3Options = Mode3Options()
    mode4: Mode4Options = Mode4Options()

    def get_kwargs(self, mode:int):
        """ Extract the options for a specific mode as a dictionary
            (for use as kwargs)."""
        if mode == 0:
            return dict()
        elif mode == 3:
            return asdict(self.mode3)
        elif mode == 4:
            return asdict(self.mode4)
        else:
            raise ValueError(f"Unsupported mode: {mode}.")