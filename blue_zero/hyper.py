from dataclasses import dataclass
from argparse_dataclass import ArgumentParser


__all__ = ['HyperParams', 'hp_parser']


@dataclass
class HyperParams:
    ############################
    # EMBEDDING/Q NET
    ############################

    # number of convolutional layers (embedding step)
    depth: int = 5

    # dimension of space in which to embed each board square
    num_feat: int = 32

    # number of hidden neurons between inputs (embedded state) and outputs
    # (q value for each square)
    num_hidden: int = 128

    # size of 2D convolution kernel
    kernel_size: tuple = (3, 3)

    # initialize all network weights uniformly between (-w_scale, w_scale)
    w_scale = 0.01

    ############################
    # TRAINING
    ############################

    # learning rate
    lr: float = 1e-4

    # how many boards to use per stochastic gradient descent iteration
    batch_size: int = 64

    # maximum number of training iterations to perform
    max_epochs: int = 10 ** 5

    # every this many epochs, have agent play through some training examples
    # and store in replay buffer
    play_freq: int = 100   # Rate at which to play envs

    # number of environments from train set to play through/store each time
    num_play: int = 100

    ############################
    # ANNEALING
    ############################

    # initial random move probability for greedy epsilon strategy
    eps_start: float = 1.0  #

    # final random move probability for greedy epsilon strategy
    eps_end: float = 0.01  # final prob. to pick random move (greedy-epsilon)

    # number of epochs over which to linearly decay eps from eps_start to
    # eps_end
    eps_decay_time: float = 10 ** 4

    # method used to update target net from policy net
    # one of ['hard', 'soft']
    target_update_mode: str = 'hard'

    # hard update the target network every this many epochs
    # (only used if target_update_mode = 'hard')
    hard_update_freq: int = 1000

    # rate at which to soft update target network parameters every epoch
    # (only used if target_update_mode = 'soft')
    soft_update_rate: float = 0.0001

    ############################
    # FITTING
    ############################

    # discount factor for future rewards
    gamma: float = 1.0

    # clip the magnitudes of parameter gradients to this value
    # before backward pass
    max_grad_norm: float = 1.0

    ############################
    # REPLAY MEMORY
    ############################

    # capacity of replay buffer
    mem_size: int = 10 ** 6

    # number of complete environments to play through/store before training
    num_burn_in: int = 100

    # the "N" in "N-Step" replay memory
    step_diff: int = 1


hp_parser = ArgumentParser(HyperParams)
