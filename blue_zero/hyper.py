from dataclasses import dataclass

__all__ = ['HyperParams']


@dataclass
class HyperParams:
    # EMBEDDING/Q NET
    depth: int = 5  # total number of convolutional layers
    num_feat: int = 32  # no. of node features (= no. of conv. filters)
    num_hidden: int = 128   # size of hidden layer between inputs (embedded node
                           # feats. + pools of board state) and final output
                           # for each node
    kernel_size: tuple = (3, 3)  # size of kernel for convolution
    w_scale = 0.01  # initial network weights drawn from
                    # uniform(-w_scale, w_scale)

    # TRAINING
    gamma: float = 1.0  # discount factor
    lr: float = 1e-4    # learning rate
    batch_size: int = 64  # num. of states to use per training iteration
    max_epochs: int = 10 ** 5  # number of trains to do before terminating
    play_freq: int = 100   # Rate at which to play envs
    num_play: int = 100   # Number of environments to play at every "play" step
    train_set_size = 100_000  # Number of random grids to use for training

    # ANNEALING
    eps_start: float = 1.0  # start prob. to pick random move (greedy epsilon)
    eps_end: float = 0.01  # final prob. to pick random move (greedy-epsilon)
    eps_decay_time: float = 10 ** 4  # decay eps over this many epochs
    hard_update_freq: int = 1000      # 'hard' update target net every this
                                      # many epochs
    soft_update_rate: float = 0.0001  # 'soft' update rate for target net
    target_update_mode: str = 'hard'  # one ['hard', 'soft']; method used to
                                      # updat target net from policy net

    # FITTING/LOSS FUNCTION
    max_grad_norm: float = 1.0  # clip the magnitudes of gradients to this value
                                # before backward pass

    # MEMORY
    mem_size: int = 10 ** 6  # max no. of transitions to store in replay
    num_burn_in: int = 100   # no. of environments to play and store before
                              # starting training
    step_diff: int = 1  # no. of steps to use between initial/target states

    # VALIDATION
    validation_freq: int = 100  # perform validation every this many epochs
    validation_set_size = 100  # Number of random grids to use for validation

