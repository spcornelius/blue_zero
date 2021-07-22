from dataclasses import dataclass
from simple_parsing import choice, Serializable


__all__ = ['HyperParams', 'TrainParams']


@dataclass
class TrainParams(Serializable):
    """ parameters for double-Q learning with N-step replay memory """

    # minibatch size
    batch_size: int

    # maximum number of training iterations to perform
    max_epochs: int

    # play through/store training examples every this many epochs
    play_freq: int

    # number of training examples to play each time
    num_play: int

    # number of epochs to anneal exploration parameter
    anneal_epochs: int

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

    # how often to record network snapshots / losses
    snapshot_freq: int = 0

    # exploration strategy
    exploration: str = choice('eps_greedy', 'softmax')

    # if false, don't anneal exploration parameter from start to final
    # value
    anneal: bool = True

    # initial random move probability for greedy epsilon strategy
    eps_max: float = 1.0

    # final random move probability for greedy epsilon strategy
    eps_min: float = 0.01

    # initial temperature for Boltzmann kernel
    T_max: float = 100.0

    # final (asymptotic) temperature for Boltzmann kernel
    T_min: float = 1.0


@dataclass
class HyperParams(Serializable):
    """ Parameters representing a complete training run """
    qnet: dict
    mode: dict
    replay: dict
    training: TrainParams
