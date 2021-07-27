from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from blue_zero.agent import QAgent, EpsGreedyQAgent, SoftMaxQAgent
from blue_zero.mode import BlueMode
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
                 train_set: Iterable[BlueMode],
                 validation_set: Iterable[BlueMode],
                 memory: NStepReplayMemory,
                 p: TrainParams,
                 device='cpu'):
        self.policy_net = net
        self.policy_net.to(device)
        self.target_net = deepcopy(self.policy_net)

        self.train_set = tuple(train_set)
        self.validation_set = tuple(validation_set)

        self.memory = memory

        weights = self.policy_net.parameters()
        opt_kwargs = p.optimizer
        opt_name = opt_kwargs.pop('name').lower()
        try:
            self.optimizer = optimizers[opt_name](weights, **opt_kwargs)
        except KeyError:
            raise ValueError(f"Unrecognized optimizer '{opt_name}'.")
        self.loss_func = torch.nn.SmoothL1Loss()

        self.epoch = 0
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

        self.snapshots = {}
        self.snapshot_times = self._get_snapshot_times(p)

    @staticmethod
    def _get_snapshot_times(p: TrainParams):
        t1 = np.arange(0, p.snapshot_all_before)
        t2 = np.logspace(np.log10(p.snapshot_all_before),
                         np.log10(p.max_epochs),
                         p.num_addl_snapshots, dtype=np.int)
        t = np.hstack((t1, t2))
        t = np.insert(t, 0, 0)
        return np.sort(np.unique(t))

    def snapshot_if_required(self) -> None:
        # idx of next scheduled snapshot time
        i = np.searchsorted(self.snapshot_times, self.epoch)
        t = self.snapshot_times[i]
        if self.epoch == t:
            self.snapshots[t] = deepcopy(self.policy_net.state_dict())

    @property
    def eps(self) -> float:
        """ Current value of epsilon (random action probability). """
        e1, e2, t = self.p.eps_max, self.p.eps_min, self.p.anneal_epochs
        if self.p.anneal:
            return e2 + max(0.0, (e1 - e2) * (t - self.epoch) / t)
        else:
            return e2

    @property
    def T(self) -> float:
        """ Current value of T (temperature for Boltzmann kernel). """
        p = self.p
        t = self.epoch
        if p.anneal:
            return p.T_min + np.exp(-t / p.anneal_epochs) * (p.T_max - p.T_min)
        else:
            return p.T_min

    def create_agent(self, greedy: bool = False):
        if greedy:
            return QAgent(self.policy_net)

        exploration = self.p.exploration
        if exploration == 'eps_greedy':
            return EpsGreedyQAgent(self.policy_net, self.eps)
        elif exploration == 'softmax':
            return SoftMaxQAgent(self.policy_net, self.T)
        else:
            raise ValueError(
                f"Unrecognized exploration strategy '{exploration}'.")

    def update_status(self, loss, perf):
        self.status_pbar.postfix[0] = loss
        self.status_pbar.postfix[1] = perf
        self.status_pbar.refresh()

    def train(self):
        p = self.p

        print()

        self.burn_in()

        loss = np.nan
        self.train_pbar.refresh()

        while self.epoch < p.max_epochs:
            # update target network from policy network if necessary, using
            # the specified approach
            self.update_target_if_required()

            if self.epoch % p.play_freq == 0:
                self.rollout()

            if self.epoch % p.validation_freq == 0:
                perf = self.validate()
                self.update_status(loss, perf)

            # snapshot the network *before* adjusting weights (so that
            # snapshot 0 is the virgin network prior to any training)
            self.snapshot_if_required()

            # do one fit iteration
            loss = self.fit()
            self.train_pbar.update(1)
            self.train_pbar.refresh()

            self.epoch += 1

        return deepcopy(self.policy_net.state_dict()), self.snapshots

    def update_target_if_required(self):
        if self.p.target_update_mode == 'soft':
            self._soft_update_target()
        elif self.p.target_update_mode == 'hard' and \
                self.epoch % self.p.hard_update_freq == 0:
            self._hard_update_target()

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

        # get q estimate using POLICY net
        q_prev = self.policy_net(s_prev, a=a)

        q = torch.zeros_like(q_prev)

        # get reward-to-go of next state according to target net
        # (but choosing the corresponding optimal action using the policy net)
        agent = self.create_agent(greedy=True)
        a_next = agent.get_action(s[~terminal])
        q[~terminal] = self.target_net(s[~terminal], a=a_next).detach()

        # update weights
        self.optimizer.zero_grad()
        loss = self.loss_func((self.p.gamma ** dt) * q + r, q_prev)
        loss.backward()
        if self.p.clip_gradients:
            clip_grad_norm_(self.policy_net.parameters(), self.p.max_grad)
        self.optimizer.step()

        return loss.detach()

    def play(self,
             envs: Iterable[BlueMode],
             pbar: tqdm = None,
             greedy: bool = False,
             memorize: bool = False):

        agent = self.create_agent(greedy=greedy)

        pbar.reset()
        for e in envs:
            e.reset()

        agent.play(envs, batch_size=self.p.batch_size, pbar=pbar)

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
        self.play(envs, pbar=pbar, memorize=True)
        pbar.close()

    def rollout(self) -> None:
        """ Play through num_play complete environments drawn from the training
            set using an epsilon-greedy strategy, then store completed envs
            in replay memory. """
        envs = np.random.choice(self.train_set, self.p.num_play, replace=False)
        self.policy_net.train()
        self.play_pbar.reset()
        self.play(envs, pbar=self.play_pbar, memorize=True)
        self.play_pbar.clear()

    def validate(self) -> float:
        """ Validate current policy net.

        Returns:
            Average solution quality over the set of validation graphs.
            Solution quality is defined by the environment in question.
        """
        self.policy_net.eval()
        self.validate_pbar.reset()
        envs = self.play(self.validation_set, greedy=True,
                         pbar=self.validate_pbar)
        self.validate_pbar.clear()

        return np.mean([e.sol_size for e in envs]).item()
