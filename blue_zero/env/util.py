from blue_zero.env.mode0 import BlueMode0
from blue_zero.env.mode3 import BlueMode3
from dataclasses import dataclass, asdict
from simple_parsing import choice


__all__ = []
__all__.extend(['env_cls', 'ModeOptions', 'Mode3Options'])


env_cls = {
    0: BlueMode0,
    3: BlueMode3
}


@dataclass
class Mode3Options:
    """ options for mode 3 """
    # direction of current (mode 3 only)
    direction: str = choice('horizontal', 'vertical', 'both',
                            default='horizontal')


@dataclass
class ModeOptions:
    """" options for specific BLUE environments """
    mode3: Mode3Options = Mode3Options()

    def get_kwargs(self, mode:int):
        """ Extract the options for a specific mode as a dictionary
            (for use as kwargs)."""
        if mode == 0:
            return dict()
        elif mode == 3:
            return asdict(self.mode3)
        else:
            raise ValueError(f"Unsupported mode: {mode}.")