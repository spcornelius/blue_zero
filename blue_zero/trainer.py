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
              'rmsprop': optim.RMSprop}


class Trainer(object):
    """ Trains an Agent via stochastic gradient descent using Double-Q learning
        Use an epsilon-greedy strategy to balance exploration vs. exploitation,
        with a linear decay in epsilon. """

    def __init__(self, net: QNet,
                 train_set: Iterable[BlueEnv],
                 validation_set: Iterable[BlueEnv],
                 p: TrainParams,
                 device='cpu'):

        super().__init__()

        self.policy_net = net
        self.policy_net.to(device)
        self.policy_net.train()

        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()

        self.agent = Agent(self.policy_net)

        self.train_set = list(train_set)
        self.validation_set = list(validation_set)

        self.memory = NStepReplayMemory(p.mem_size, p.step_diff, device=device)

        weights = self.policy_net.parameters()
        try:
            self.optimizer = optimizers[p.optimizer.lower()](weights, lr=p.lr)
        except KeyError:
            raise ValueError(f"Unrecognized optimizer '{p.optimizer}'.")
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

    @property
    def eps(self) -> float:
        """ Current value of epsilon (random action probability). """
        e1, e2, t = self.p.eps_start, self.p.eps_end, self.p.eps_decay_time
        return e2 + max(0.0, (e1 - e2) * (t - self.epoch) / t)

    def train(self) -> QNet:
        p = self.p

        print()

        self.burn_in()

        loss = np.nan
        self.train_pbar.refresh()
        while self.epoch < p.max_epochs:
            if self.epoch % p.play_freq == 0:
                self.play_pbar.reset()
                self.play_games(p.num_play, pbar=self.play_pbar)
                self.play_pbar.clear()

            if self.epoch % p.validation_freq == 0:
                perf = self.validate()
                self.status_pbar.postfix[0] = loss
                self.status_pbar.postfix[1] = perf
                self.status_pbar.refresh()

            # do one fit iteration
            loss = self.fit()
            self.epoch += 1
            self.train_pbar.update(1)
            self.train_pbar.refresh()

            # update target network from policy network if necessary, using
            # the specified approach
            if p.target_update_mode == 'soft':
                self._soft_update_target()
            elif p.target_update_mode == 'hard' and \
                    self.epoch % p.hard_update_freq == 0:
                self._hard_update_target()

        return self.agent.net

    def _soft_update_target(self) -> None:
        tau = self.p.soft_update_rate
        # soft update target network
        for p, p_target in zip(self.policy_net.parameters(),
                               self.target_net.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)
        self.target_net.eval()

    def _hard_update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def burn_in(self):
        pbar = tqdm(total=self.p.num_burn_in, position=0,
                    desc="Burning in", unit=" games",
                    bar_format=pbar_format, leave=False)
        self.play_games(self.p.num_burn_in, pbar=pbar)
        pbar.close()

    def fit(self) -> float:
        """ Perform one optimization step, sampling a batch of transitions
            from replay memory and performing stochastic gradient descent.
        """
        self.policy_net.train()
        self.target_net.eval()

        s_prev, a, s, dr, terminal = self.memory.sample(self.p.batch_size)

        # get reward-to-go from state s from TARGET net
        with torch.no_grad():
            q = self.target_net(s).view(self.p.batch_size, -1).amax(1)

        # terminal states by definition have zero reward-to-go
        q[terminal] = 0.0

        # shouldn't be necessary as the optimizer only knows about the policy
        # net's parameters, but make sure the target q values don't contribute
        # to the gradient (target network is fixed in Dueling DQN)
        q = q.detach()

        # get q estimate using POLICY net
        q_prev = self.policy_net(s_prev, a=a)

        # update weights
        self.optimizer.zero_grad()
        loss = self.loss_func(q_prev, dr + self.p.gamma * q)
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), self.p.max_grad_norm)
        self.optimizer.step()

        return loss.detach()

    def play_games(self, n: int, pbar: tqdm = None) -> None:
        """ Play through n complete environments drawn from the training
            set using an epsilon-greedy strategy, then
            store completed envs in replay _memory. """
        envs = np.random.choice(self.train_set, n, replace=False)

        for e in envs:
            e.reset()

        self.agent.play(envs, eps=self.eps, pbar=pbar, device=self.device)

        for e in envs:
            assert e.done
            self.memory.store(e)

    def validate(self):
        """ Validate current policy net.

        Returns:
            Average solution quality over the set of validation graphs.
            Solution quality is defined by the environment in question.
        """
        for e in self.validation_set:
            e.reset()

        self.validate_pbar.reset()

        # use a full greedy strategy (eps=0) for validation
        self.agent.play(self.validation_set,
                        eps=0, pbar=self.validate_pbar,
                        device=self.device)
        self.validate_pbar.clear()

        return np.mean([e.sol_size for e in self.validation_set])
