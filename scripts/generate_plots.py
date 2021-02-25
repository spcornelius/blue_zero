import sys
from dataclasses import dataclass, asdict
from time import sleep
import matplotlib.pyplot as plt

import pygame
from path import Path
from simple_parsing import ArgumentParser, field

from blue_zero.agent import Agent
from blue_zero.env import ModeOptions
from blue_zero.net.dqn import DQN
from blue_zero.env.util import env_cls


@dataclass
class Options:
    # .pt file containing a trained model
    file: Path = field(alias='-f', required=True)

    # size of board to play on
    n: int = field(alias='-n', required=True)

    # green probability
    # p: float = field(alias='-p', required=True)

    # game mode
    mode: int = field(required=True)

    # time delay between taking moves
    pause: float = 0.2

def play_envs(envs: list, agent: Agent, random: bool):
    moves = []
    
    epsilon = 0
    if random: epsilon = 1

    # playing each env with the agent specified
    for env in temp_envs:
        actions = 0
        while True:
            if not env.done:
                a = agent.get_action(env.state, eps=epsilon)
                env.step(a)
                actions += 1
            else:
                env.reset()
                break

        moves.append(actions)

    return moves

def generate_envs(n: int, p: float, mode: int, num_envs: int):
    envs = []
    index = 0

    # randomly initializing board states
    while index < num_envs:
        env = env_cls[mode].from_random((n, n), p, with_gui=False, **kwargs)
        if env.done:
            continue
        else:
            index += 1
            envs.append(env)

    return env

def plot_lineplot(random_moves, agent_moves):

    mean_random_moves = [np.mean(value) for key, value in random_moves.items()]
    mean_agent_moves = [np.mean(value) for key, value in agent_moves.items()]

    x_ticks = random_moves.keys()

    # Plotting a lineplot of mean of different trained agent's moves with their respective random agent's moves
    plt.plot(x_ticks, mean_random_moves, label="random agent")
    plt.plot(x_ticks, mean_agent_moves, label="trained agent")
    plt.title("Comparison of average moves of agents over different probability values")
    plt.xlabel("Probability value")
    plt.ylabel("Average moves")
    plt.legend()
    plt.show()

def main2(file: Path, n: int, mode: int,
         pause: float = 0.2, **kwargs):

    pygame.init()
    num_envs = 1000

    p_values = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    random_moves_dict = {}
    agent_moves_dict = {}

    # iterating over different p values and simulating 1000 games
    # each for both random and trained agent
    for p in p_values:
        agent_path = file + '/p_'+ str(p) + '/mode3_model.pt'
        net = DQN.load(agent_path)
        agent = Agent(net)

        # generating randomly initialized board games
        envs = generate_envs(n, p, mode, num_envs)

        env_actions = []

        # simulation of random agent playing
        temp_envs = envs.copy()
        moves_random = play_envs(temp_envs, agent, random=True)
        random_moves_dict[p] = moves_random
  
        # simulation of trained agent playing
        temp_envs = envs.copy()
        moves_agent = play_envs(temp_envs, agent, random=False)
        agent_moves_dict[p] = moves_agent

        print("Done with the p value: "+str(p))

    # saving the moves data to files
    np.save('random_moves.npy', random_moves_dict)
    np.save('agent_moves.npy', agent_moves_dict)

    # plotting the agent comparision
    plot_lineplot(random_moves_dict, agent_moves_dict)  


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    parser.add_arguments(ModeOptions, "mode_options")
    args = parser.parse_args()
    kwargs = asdict(args.options)
    kwargs.update(args.mode_options.get_kwargs(args.options.mode))
    main(**kwargs)
