from dataclasses import dataclass
from simple_parsing import choice, Serializable


__all__ = ['NetParams', 'TrainParams', 'HyperParams']


@dataclass
class NetParams(Serializable):
    """ parameters for deep Q network """

    # number of convolutional layers in embedding
    depth: int

    # dimension of embedding
    num_feat: int

    # number of hidden neurons between embedding and outputs
    num_hidden: int

    # size of embedding convolution kernel
    kernel_size: int

    # include bias terms in convolutions
    with_conv_bias: bool


@dataclass
class TrainParams(Serializable):
    """ parameters for double-Q learning with N-step replay memory """

    # learning rate
    lr: float

    # number of replay transitions to fit during each optimization step
    batch_size: int

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

    # size of replay buffer
    mem_size: int

    # number of complete environments to play through/store before training
    num_burn_in: int

    # discount factor for future rewards
    gamma: float

    # clip gradient norms to no more than this value
    max_grad_norm: float

    # the "N" in "N-Step" replay memory
    step_diff: int

    # play through validation data every this many epochs
    validation_freq: int

    # optimizer to use
    optimizer: str = choice('adam', 'sgd', 'rmsprop')

    # method used to update target net from policy net
    target_update_mode: str = choice('hard', 'soft')


@dataclass
class HyperParams(Serializable):
    """ Combined hyperparameters (network + training) representing a
        single training run. """

    net_params: NetParams
    train_params: TrainParams
