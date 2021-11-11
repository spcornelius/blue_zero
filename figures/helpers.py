from itertools import product as iterproduct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import colors

from blue_zero import config as cfg
from blue_zero.agent import QAgent
from blue_zero.mode import mode_registry
from blue_zero.mode.base import BlueMode
from blue_zero.qnet import QNet

blue_colors = [
    cfg.black,
    cfg.green,
    cfg.blue,
    cfg.orange,
]
blue_colors = np.array(blue_colors) / 255
cm = colors.ListedColormap(blue_colors)
norm = colors.BoundaryNorm(boundaries=[1, 2, 3, 4, 5], ncolors=cm.N)


def get_net(trained_mode=0, n=20, models_root=".."):
    trained_file = models_root + f"/mode{trained_mode}/{n}/trained_model.pt"
    net = QNet.load(trained_file)
    return net


def get_env(game_mode=0, n=20, p=0.8) -> BlueMode:
    env = mode_registry[game_mode].from_random((n, n), p)
    return env


def get_q(net, env):
    q = net(env.state).detach().numpy().squeeze()
    n = q.shape[0]
    return abs(q * n)


def plot_modes():
    net0 = get_net(0)
    net3 = get_net(3)
    net4 = get_net(4)
    env = get_env(p=0.65)
    q0 = get_q(net0, env)
    q3 = get_q(net3, env)
    q4 = get_q(net4, env)
    # current_cmap = matplotlib.cm.get_cmap()
    # current_cmap.set_bad(color='black')
    f, ax = plt.subplots(1, 3, figsize=(9, 4))
    ax[0].imshow(q0)
    ax[0].set_title("network")
    ax[1].imshow(q3)
    ax[1].set_title("flow")
    ax[2].imshow(q4)
    ax[2].set_title("noodle")
    [a.set_xticks([]) for a in ax]
    [a.set_yticks([]) for a in ax]


def get_played_envs(trained_mode, game_mode, n, p, n_games, models_root):
    net = get_net(trained_mode=trained_mode, n=n, models_root=models_root)
    envs = [get_env(game_mode, n, p) for _ in range(n_games)]
    QAgent(net).play(envs)
    return envs


def get_steps_df(n=15, p=0.8, games=1000, models_root="/app/"):
    results = {}
    for trained_mode, game_mode in iterproduct((0, 3), (0, 3)):
        envs = get_played_envs(
            trained_mode, game_mode, n=n, p=p, n_games=games, models_root=models_root
        )
        steps = [e.steps_taken for e in envs]
        results[(trained_mode, game_mode)] = steps
    steps_df = pd.DataFrame(results)
    steps_df.columns.names = ["trained_on", "played_on"]
    return steps_df


def do_plot(a, b, steps_df):
    sb.kdeplot(x=(a, b), data=steps_df, bw_adjust=1.4, cut=0, label=f"{a} on {b}")
    plt.gca().set_xlabel("steps to completion")
    plt.legend()


def do_on_task_plot(steps_df):
    plt.figure(figsize=(5, 5))
    do_plot(0, 0, steps_df)
    do_plot(3, 3, steps_df)
    plt.savefig("on_task.png")


def do_off_task_plot(steps_df):
    plt.figure(figsize=(5, 5))
    do_plot(0, 3, steps_df)
    do_plot(3, 0, steps_df)
    plt.savefig("off_task.png")


def plot_six_panel_game(env):
    assert len(env.states) == 5
    arrs = [x.T @ [1, 2, 3, 4] for x in env.states] + [env.state.T @ [1, 2, 3, 4]]
    f, ax = plt.subplots(2, 3)
    for arr, subax in zip(arrs, ax.flatten().squeeze()):
        subax.imshow(arr, cmap=cm, norm=norm)
        subax.set_xticks([])
        subax.set_yticks([])
    return f
