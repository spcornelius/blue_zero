import sys
from time import sleep

import numpy as np
import pygame
import torch
from path import Path

from blue_zero.agent import Agent
from blue_zero.env.blue import Blue
from blue_zero.hyper import HyperParams
from blue_zero.net.dqn import DQN

np.random.seed(0)
torch.random.manual_seed(0)

n = 40
p = 0.65
device = 'cuda:1'
idx = 1

untrained_model_save_file = Path(f"./saved_models/untrained_model_10.pt")
trained_model_save_file = Path(f"./saved_models/trained_model_10.pt")
# validation_data_save_file = Path(f"./validation_data/validation_data_{
# n}.npy")

hp = HyperParams()
net = DQN(hp.num_feat, hp.num_hidden, hp.depth,
          kernel_size=hp.kernel_size)

smart_net = DQN(hp.num_feat, hp.num_hidden, hp.depth,
                kernel_size=hp.kernel_size)
smart_net.load_state_dict(torch.load(trained_model_save_file))
smart_net.eval()
smart_agent = Agent(smart_net)

dumb_net = DQN(hp.num_feat, hp.num_hidden, hp.depth,
               kernel_size=hp.kernel_size)
dumb_net.load_state_dict(torch.load(untrained_model_save_file))
dumb_net.eval()
dumb_agent = Agent(dumb_net)

# validation_set = np.load(validation_data_save_file)
# state = validation_set[idx].squeeze()
env = Blue.from_random((n, n), p, with_gui=True)

event = pygame.event.wait()
done = False

sleep(5.0)
dumb_agent.play_envs([env], pause=0.2)
sleep(5.0)
env.reset()
sleep(2.0)
smart_agent.play_envs([env], pause=0.2)

while True:
    event = pygame.event.wait()
    if event.type == pygame.QUIT:
        sys.exit(0)
