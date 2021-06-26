from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from blue_zero.agent import Agent
from blue_zero.env import BlueEnv
from blue_zero.params import TrainParams
from blue_zero.qnet import QNet
from blue_zero.replay import NStepReplayMemory

__all__ = []
__all__.extend([
    'Trainer'
])

BOLD = '\033[1m'
END = '\033[0m'

l_bar = '{desc}: {percentage:3.0f}%|'
r_bar = '| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
pbar_format = f"{l_bar}{{bar}}{r_bar}"

optimizers = {'adam': optim.Adam,
              'adamw': optim.AdamW,
              'sgd': optim.SGD,
              'rmsprop': optim.RMSprop,
              'adadelta': optim.Adadelta}


class Trainer(object):
    """ Trains an Agent via stochastic gradient descent using Double-Q learning
        Use an epsilon-greedy strategy to balance exploration vs. exploitation,
        with a linear decay in epsilon. """

    def __init__(self, net: QNet,
                 train_set: Iterable[BlueEnv],
                 validation_set: Iterable[BlueEnv],
                 memory: NStepReplayMemory,
                 p: TrainParams,
                 device='cpu'):

        super().__init__()

        self.policy_net = net
        self.policy_net.to(device)
        self.target_net = deepcopy(self.policy_net)
        self.agent = Agent(self.policy_net)

        self.train_set = list(train_set)
        self.validation_set = list(validation_set)

        self.memory = memory

        weights = self.policy_net.parameters()
        opt_kwargs = p.optimizer
        opt_name = opt_kwargs.pop('name').lower()
        try:
            Opt = optimizers[opt_name]
        except KeyError:
            raise ValueError(f"Unrecognized optimizer '{opt_name}'.")
        self.optimizer = Opt(weights, **opt_kwargs)
        self.loss_func = torch.nn.MSELoss()

        self.epoch = 0
        self.best_epoch = None
        self.best_perf = np.inf
        self.best_params = None
        self.p = p
        self.device = device

        self.train_pbar = tqdm(total=p.max_epochs, position=0,
                               desc="Training", unit=" epochs",
                               leave=False)
        self.status_pbar = tqdm(position=2, total=0,
                                bar_format=f"{BOLD}Loss: {{postfix[0]:1.2e}}"
                                           f"  |  "
                                           f"Avg. moves to win: "
                                           f"{{postfix[1]:.2f}}{END}",
                                postfix=[np.nan, np.nan])
        self.play_pbar = tqdm(total=p.num_play, position=1,
                              desc="    Playing", unit=" games",
                              bar_format=pbar_format)
        self.validate_pbar = tqdm(total=len(self.validation_set), position=1,
                                  desc="    Validating", unit=" games",
                                  bar_format=pbar_format, leave=False)
        self.train_pbar.clear()
        self.status_pbar.clear()
        self.play_pbar.clear()
        self.validate_pbar.clear()

        self.losses = {}
        self.snapshots = {}

    @property
    def eps(self) -> float:
        """ Current value of epsilon (random action probability). """
        e1, e2, t = self.p.eps_start, self.p.eps_end, self.p.eps_decay_time
        return e2 + max(0.0, (e1 - e2) * (t - self.epoch) / t)

    def train(self):
        p = self.p

        print()

        self.burn_in()

        loss = np.nan
        self.train_pbar.refresh()

        while self.epoch < p.max_epochs:
            if self.epoch % p.play_freq == 0:
                self.play_pbar.reset()
                self.play_games()
                self.play_pbar.clear()

            if self.epoch % p.validation_freq == 0:
                self.validate_pbar.reset()
                perf = self.validate()
                self.validate_pbar.clear()
                self.status_pbar.postfix[0] = loss
                self.status_pbar.postfix[1] = perf
                self.status_pbar.refresh()

            # snapshot network *before* adjusting weights
            try:
                if self.epoch % p.snapshot_freq == 0:
                    self.snapshots[self.epoch] = deepcopy(self.policy_net)
            except ZeroDivisionError:
                pass

            # do one fit iteration
            loss = self.fit()
            self.train_pbar.update(1)
            self.train_pbar.refresh()

            self.losses[self.epoch] = loss

            # update target network from policy network if necessary, using
            # the specified approach
            if p.target_update_mode == 'soft':
                self._soft_update_target()
            elif p.target_update_mode == 'hard' and \
                    self.epoch % p.hard_update_freq == 0:
                self._hard_update_target()

            self.epoch += 1

        return self.policy_net, self.snapshots, self.losses

    def _soft_update_target(self) -> None:
        tau = self.p.soft_update_rate
        # soft update target network
        for p, p_target in zip(self.policy_net.parameters(),
                               self.target_net.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)

    def _hard_update_target(self) -> None:
        state_dict = deepcopy(self.policy_net.state_dict())
        self.target_net.load_state_dict(state_dict)

    def fit(self) -> float:
        """ Perform one optimization step, sampling a batch of transitions
            from replay memory and performing stochastic gradient descent.
        """
        s_prev, a, s, r, terminal, dt = self.memory.sample(self.p.batch_size)

        # get reward-to-go of next state according to target net
        # (but choosing the corresponding optimal action using the policy net)
        a_next = self.agent.get_action(s, eps=0)
        q = self.target_net(s, a=a_next).detach()

        # terminal states by definition have zero reward-to-go
        q[terminal] = 0.0

        # get q estimate using POLICY net
        q_prev = self.policy_net(s_prev, a=a)

        # update weights
        self.optimizer.zero_grad()
        loss = self.loss_func(q + r, q_prev)
        loss.backward()
        if self.p.clip_gradients:
            clip_grad_norm_(self.policy_net.parameters(), self.p.max_grad)
        self.optimizer.step()

        return loss.detach()

    def play(self,
             envs: Iterable[BlueEnv],
             eps: float = None,
             pbar: tqdm = None,
             rotate: bool = True,
             memorize: bool = False):

        if eps is None:
            eps = self.eps

        pbar.reset()
        for e in envs:
            e.reset()
            if rotate:
                k = np.random.randint(0, 4)
                e.state = np.ascontiguousarray(
                    np.rot90(e.state, k=k, axes=(1, 2)))

        self.agent.play(envs, batch_size=self.p.batch_size,
                        eps=eps, pbar=pbar)

        if memorize:
            for e in envs:
                assert e.done
                self.memory.store(e, self.p.gamma)

        return envs

    def burn_in(self):
        pbar = tqdm(total=self.p.num_burn_in, position=0,
                    desc="Burning in", unit=" games",
                    bar_format=pbar_format, leave=False)
        envs = np.random.choice(self.train_set, self.p.num_burn_in,
                                replace=False)
        self.policy_net.train()
        self.play(envs, eps=self.eps, pbar=pbar, memorize=True)
        pbar.close()

    def play_games(self) -> None:
        """ Play through num_play complete environments drawn from the training
            set using an epsilon-greedy strategy, then store completed envs
            in replay memory. """
        envs = np.random.choice(self.train_set, self.p.num_play, replace=False)
        self.policy_net.train()
        self.play(envs, pbar=self.play_pbar, eps=self.eps, memorize=True)

    def validate(self):
        """ Validate current policy net.

        Returns:
            Average solution quality over the set of validation graphs.
            Solution quality is defined by the environment in question.
        """
        self.policy_net.eval()
        envs = self.play(self.validation_set, eps=0, pbar=self.validate_pbar)
        return np.mean([e.sol_size for e in envs])
