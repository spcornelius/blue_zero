from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim
from gym import Env
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tqdm.std import Bar

from blue_zero.agent import Agent
from blue_zero.params import TrainParams
from blue_zero.replay import NStepReplayMemory

__all__ = []
__all__.extend([
    'Trainer'
])

BOLD = '\033[1m'
END = '\033[0m'

l_bar = '{desc}: {percentage:3.0f}%|'
r_bar = '| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
format = f"{l_bar}{{bar}}{r_bar}"


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

        self.train_pbar = tqdm(total=p.max_epochs, position=0,
                               desc="Training", unit=" epochs",
                               leave=False)
        self.log_pbar = tqdm(position=2, total=0,
                             bar_format='{desc}')
        self.play_pbar = tqdm(total=p.num_play, position=1,
                              desc="    Playing", unit=" games",
                              bar_format=format)
        self.validate_pbar = tqdm(total=len(self.validation_set), position=1,
                                  desc="    Validating", unit=" games",
                                  bar_format=format, leave=False)
        self.train_pbar.clear()
        self.log_pbar.clear()
        self.play_pbar.clear()
        self.validate_pbar.clear()

    @property
    def eps(self):
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
                self.play_games(p.num_play, pbar=self.play_pbar)
                self.play_pbar.clear()

            if self.epoch % p.validation_freq == 0:
                perf = self.validate()
                self.log_pbar.set_description_str(
                    f"{BOLD}Loss: {loss:1.2e}  |  "
                    f"Avg. moves to win: {perf:.2f}{END}"
                )

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
        for p, p_target in zip(self.agent.net.parameters(),
                               self.target_agent.net.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)
        self.target_agent.net.eval()

    def _hard_update_target(self) -> None:
        self.target_agent.net.load_state_dict(self.agent.net.state_dict())
        self.target_agent.eval()

    def burn_in(self):
        pbar = tqdm(total=self.p.num_burn_in, position=0,
                    desc="Burning in", unit=" games",
                    bar_format=format, leave=False)
        self.play_games(self.p.num_burn_in, pbar=pbar)
        pbar.close()

    def fit(self) -> float:
        """ Perform one optimization step, sampling a batch of transitions
            from replay memory and performing stochastic gradient descent.
        """
        s_prev, a, s, dr, terminal = self.memory.sample(self.p.batch_size)

        # get reward-to-go from state s from TARGET qnet
        _, q = self.target_agent.get_action(s, return_q=True)

        # terminal states by definition have zero reward-to-go
        q.masked_fill_(terminal, 0.0)

        # shouldn't be necessary as the optimizer only knows about the policy
        # qnet's parameters, but make sure the target q values don't contribute
        # to the gradient (target network is fixed in Double DuelingQNet)
        q = q.detach()

        # get past q estimate using POLICY qnet
        q_prev = self.agent.net(s_prev, a=a)

        self.optimizer.zero_grad()
        loss = self.loss_func(q_prev, dr + self.p.gamma * q)
        loss.backward()

        clip_grad_norm_(self.agent.parameters(), self.p.max_grad_norm)
        self.optimizer.step()
        return loss.detach()

    def play_games(self, n: int, pbar: Bar = None):
        """ Play through n complete environments drawn from the training
            set using an epsilon-greedy strategy, then
            store completed envs in replay _memory. """
        envs = np.random.choice(self.train_set, n, replace=False)

        for e in envs:
            e.reset()

        self.agent.play_envs(envs, pbar=pbar,
                             device=self.device)

        for e in envs:
            assert e.done
            self.memory.store(e)

    def validate(self) -> np.float:
        """ Validate current policy qnet.

        Args:
            with_pbar: Whether or not to display a progress bar
                (default `False`).

        Returns:
            Average solution quality over the set of validation graphs.
            Solution quality is defined by the environment in question.
        """
        for e in self.validation_set:
            e.reset()

        self.validate_pbar.reset()

        # use a full greedy strategy (eps=0) for validation
        self.agent.play_envs(self.validation_set,
                             eps=0, pbar=self.validate_pbar,
                             device=self.device)
        self.validate_pbar.clear()

        return np.mean([e.sol_size for e in self.validation_set])
