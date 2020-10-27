import numpy as np
import torch
import random

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from path import Path

import blue_zero.util as util
from blue_zero.agent import Agent
from blue_zero.hyper import HyperParams
from blue_zero.net.dqn import DQN
from blue_zero.trainer import Trainer

device = 'cuda:1'
p = 0.62
n = 20

untrained_model_save_file = Path(f"./saved_models/untrained_model_{n}.pt")
trained_model_save_file = Path(f"./saved_models/trained_model_{n}.pt")
validation_data_save_file = Path(f"./validation_data/validation_data_{n}.npy")

hp = HyperParams()
net = DQN(hp.num_feat, hp.num_hidden, hp.depth,
          kernel_size=hp.kernel_size)
target_net = DQN(hp.num_feat, hp.num_hidden, hp.depth,
                 kernel_size=hp.kernel_size)

util.init_weights(net, hp.w_scale)
util.init_weights(target_net, hp.w_scale)

# save initial (untrained) model
torch.save(net.state_dict(), untrained_model_save_file)

agent = Agent(net)
target_agent = Agent(net)


def gen_playable_envs(num):
    # generate a specified no. of random environments that aren't already
    # finished
    envs = []
    while len(envs) < num:
        while (env := util.gen_random_env(n, p)).done:
            pass
        envs.append(env)
    return envs


train_set = gen_playable_envs(hp.train_set_size)
validation_set = gen_playable_envs(hp.validation_set_size)

# save validation data
np.save(validation_data_save_file,
        np.stack([e.state for e in validation_set], axis=0))

trainer = Trainer(agent, target_agent,
                  train_set, validation_set,
                  hp, device=device)

trainer.train()
torch.save(trainer.best_params, trained_model_save_file)
