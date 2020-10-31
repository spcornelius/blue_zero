from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim
from gym import Env
from torch.nn.utils import clip_grad_norm_

from blue_zero.agent import Agent
from blue_zero.hyper import TrainParams
from blue_zero.replay import NStepReplayMemory
from torch.nn import Module

__all__ = []
__all__.extend([
    'Trainer'
])


class Trainer(object):
    """ Trains an Agent via stochastic gradient descent using Double-Q learning
        Use an epsilon-greedy strategy to balance exploration vs. exploitation,
        with a linear decay in epsilon. """

    def __init__(self, net: Module,
                 train_set: Iterable[Env],
                 validation_set: Iterable[Env],
                 p: TrainParams,
                 device='cpu'):

        super().__init__()
        self.agent = Agent(net).to(device=device)
        self.target_agent = deepcopy(self.agent)
        self.agent.train()
        self.target_agent.eval()
        self.train_set = list(train_set)
        self.validation_set = list(validation_set)

        self.memory = NStepReplayMemory(p.mem_size, p.step_diff,
                                        device=device)

        if p.optimizer == 'adam':
            optimizer = optim.Adam(self.agent.parameters(), lr=p.lr)
        elif p.optimizer == 'sgd':
            optimizer = optim.SGD(self.agent.parameters(), lr=p.lr)
        elif p.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.agent.parameters(), lr=p.lr)
        else:
            raise ValueError(f"Unrecognized optimizer '{p.optimizer}'.")
        self.optimizer = optimizer
        self.loss_func = torch.nn.MSELoss()

        self.epoch = 0
        self.best_epoch = None
        self.best_perf = np.inf
        self.best_params = None
        self.p = p
        self.device = device

    @property
    def eps(self):
        """ Current value of epsilon (random action probability). """
        e1, e2, t = self.p.eps_start, self.p.eps_end, self.p.eps_decay_time
        return e2 + max(0.0, (e1 - e2) * (t - self.epoch) / t)

    def train(self):
        p = self.p

        # burn-in
        self.play_games(p.num_burn_in, with_pbar=True)

        loss = np.nan

        while self.epoch < p.max_epochs:
            if self.epoch % p.play_freq == 0:
                self.play_games(p.num_play)

            if self.epoch % p.validation_freq == 0:
                sol_sizes = self.validate()
                perf = np.mean(sol_sizes)
                print(f"Epoch: {self.epoch}, avg. sol size: "
                      f"{perf:}, loss: {loss:1.3e}.")
                if perf < self.best_perf:
                    self.best_epoch, self.best_perf = self.epoch, perf
                    best_params = deepcopy(self.agent.net.state_dict())
                    for k, v in best_params.items():
                        best_params[k] = v.cpu()
                    self.best_params = best_params

            # do one fit iteration
            loss = self.fit()
            self.epoch += 1

            if p.target_update_mode == 'soft':
                self._soft_update_target()
            elif p.target_update_mode == 'hard' and \
                    self.epoch % p.hard_update_freq == 0:
                self._hard_update_target()

    def _soft_update_target(self) -> None:
        tau = self.p.soft_update_rate
        # soft update target network
        for p, p_target in zip(self.agent.net.parameters(),
                               self.target_agent.net.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)
        self.target_agent.net.eval()

    def _hard_update_target(self) -> None:
        self.target_agent.net.load_state_dict(self.agent.net.state_dict())
        self.target_agent.eval()

    def fit(self) -> float:
        """ Perform one optimization step, sampling a batch of transitions
            from replay memory and performing stochastic gradient descent.
        """
        s_prev, a, s, dr, terminal = self.memory.sample(self.p.batch_size)

        # get reward-to-go from state s from TARGET net
        _, q = self.target_agent.get_action(s, return_q=True)

        # terminal states by definition have zero reward-to-go
        q.masked_fill_(terminal, 0.0)

        # shouldn't be necessary as the optimizer only knows about the policy
        # net's parameters, but make sure the target q values don't contribute
        # to the gradient (target network is fixed in Double DQN)
        q = q.detach()

        # get past q estimate using POLICY net
        q_prev = self.agent.net(s_prev, a=a)

        self.optimizer.zero_grad()
        loss = self.loss_func(q_prev, dr + self.p.gamma * q)
        loss.backward()

        clip_grad_norm_(self.agent.parameters(), self.p.max_grad_norm)
        self.optimizer.step()
        return loss.detach()

    def play_games(self, n: int, with_pbar: bool = False):
        """ Play through n complete environments drawn from the training
            set using an epsilon-greedy strategy, then
            store completed envs in replay _memory. """
        envs = np.random.choice(self.train_set, n, replace=False)

        for e in envs:
            e.reset()

        self.agent.play_envs(envs, eps=self.eps, with_pbar=with_pbar,
                             device=self.device)

        for e in envs:
            assert e.done
            self.memory.store(e)

    def validate(self, with_pbar: bool = False) -> np.float:
        """ Validate current policy net.

        Args:
            with_pbar: Whether or not to display a progress bar
                (default `False`).

        Returns:
            Average solution quality over the set of validation graphs.
            Solution quality is defined by the environment in question.
        """
        for e in self.validation_set:
            e.reset()

        # use a full greedy strategy (eps=0) for validation
        self.agent.play_envs(self.validation_set,
                             eps=0, with_pbar=with_pbar,
                             device=self.device)
        return np.mean([e.sol_size for e in self.validation_set])
