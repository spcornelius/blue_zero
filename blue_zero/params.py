from dataclasses import dataclass
from simple_parsing import choice, Serializable


__all__ = ['HyperParams', 'TrainParams']


@dataclass
class TrainParams(Serializable):
    """ parameters for double-Q learning with N-step replay memory """

    # maximum number of training iterations to perform
    max_epochs: int

    # play through/store training examples every this many epochs
    play_freq: int

    # number of training examples to play each time
    num_play: int

    # initial random move probability for greedy epsilon strategy
    eps_start: float

    # final random move probability for greedy epsilon strategy
    eps_end: float

    # number of epochs to go from eps_start to eps_end
    eps_decay_time: float

    # hard update the target network every this many epochs
    # (ignored unless target_update_mode = hard)
    hard_update_freq: int

    # rate at which to soft update target network parameters every epoch
    # (ignored unless target_update_mode = soft)
    soft_update_rate: float

    # number of complete environments to play through/store before training
    num_burn_in: int

    # discount factor for future rewards
    gamma: float

    # clip gradient values to no more than this in magnitude
    max_grad: float

    # play through validation data every this many epochs
    validation_freq: int

    # name of optimizer to use and optional kwargs
    optimizer: dict

    # whether to use gradient value clipping
    clip_gradients: bool = True

    # method used to update target qnet from policy qnet
    target_update_mode: str = choice('hard', 'soft')


@dataclass
class HyperParams(Serializable):
    """ Parameters representing a complete training run """
    qnet: dict
    mode: dict
    replay: dict
    training: TrainParams
