from blue_zero.env.mode0 import BlueMode0
from blue_zero.env.mode3 import BlueMode3

__all__ = []
__all__.extend(['env_cls'])


env_cls = {0: BlueMode0,
           3: BlueMode3
           }
