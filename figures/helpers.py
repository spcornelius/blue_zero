from itertools import product as iterproduct
from pathlib import Path

from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colors
import matplotlib.cm
import seaborn as sb
from tqdm import tqdm

from blue_zero import config as cfg
from blue_zero.agent import EpsGreedyQAgent, QAgent
from blue_zero.mode import mode_registry
from blue_zero.mode.base import BlueMode
from blue_zero.qnet import QNet

blue_colors = [
    cfg.black,
    cfg.green,
    cfg.blue,
    cfg.orange,
]
pretty_names = {-1: "random", 0: "network", 3: "flow"}
blue_colors = np.array(blue_colors) / 255
game_cm = colors.ListedColormap(blue_colors)
norm = colors.BoundaryNorm(boundaries=[1, 2, 3, 4, 5], ncolors=game_cm.N)

q_cm = matplotlib.cm.get_cmap("plasma_r").copy()
q_cm.set_bad(color="black")


def get_net(trained_mode=0, n=20, models_root=".."):
    trained_file = models_root + f"/mode{trained_mode}/{n}/trained_model.pt"
    net = QNet.load(trained_file)
    return net


def get_env(game_mode=0, n=20, p=0.8) -> BlueMode:
    env = mode_registry[game_mode].from_random((n, n), p)
    return env


def get_q(net, state):
    q = net(state).detach().numpy().squeeze()
    n = q.shape[0]
    return abs(q * n)


def plot_modes():
    net0 = get_net(0)
    net3 = get_net(3)
    net4 = get_net(4)
    env = get_env(p=0.65)
    q0 = get_q(net0, env.state)
    q3 = get_q(net3, env.state)
    q4 = get_q(net4, env.state)
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


def get_played_envs(trained_mode, game_mode, n, p, n_games, models_root, random=False):
    net = get_net(trained_mode=trained_mode, n=n, models_root=models_root)
    envs = [get_env(game_mode, n, p) for _ in range(n_games)]
    if random:
        EpsGreedyQAgent(net, eps=1).play(envs, pbar=True)
    else:
        QAgent(net).play(envs, pbar=True)
    return envs


def make_filename(trained_mode, game_mode, n, p):
    return f"train_{trained_mode}_play_{game_mode}_n_{n}_p_{p:.4f}"


def parse_filename(filename):
    keys = filename.split("_")[::2]
    vals = filename.split("_")[1::2]
    vals = [float(i) if "." in i else int(i) for i in vals]
    return dict(zip(keys, vals))


def run_games_and_save(
    trained_mode,
    game_mode,
    n,
    p,
    n_games,
    models_root,
    output_root,
    random=False,
):
    envs = get_played_envs(
        trained_mode, game_mode, n, p, n_games, models_root, random=random
    )
    if random:
        trained_mode = -1
    fname = Path(output_root) / make_filename(trained_mode, game_mode, n, p)
    append_game_lengths(fname, envs)


def append_game_lengths(filename, envs):
    with open(filename, "a") as f:
        f.write("\n".join([str(e.steps_taken) for e in envs]))
        f.write("\n")


def load_game_lengths(output_root):
    runs = []
    for fname in glob(str(Path(output_root) / "*")):
        with open(fname) as f:
            game_lengths = [int(i) for i in f.readlines()]
        run_data = parse_filename(fname.split("/")[-1])
        run_data["game_lengths"] = game_lengths
        runs.append(run_data)
    return runs


def get_steps_df(n=15, p=0.8, games=1000, models_root="/app/"):
    results = {}
    for trained_mode, game_mode in iterproduct((0, 3), (0, 3)):
        print(f"Trained on {trained_mode}, playing on {game_mode}")
        envs = get_played_envs(
            trained_mode, game_mode, n=n, p=p, n_games=games, models_root=models_root
        )
        steps = [e.steps_taken for e in envs]
        results[(trained_mode, game_mode)] = steps
    steps_df = pd.DataFrame(results)
    steps_df.columns.names = ["trained_on", "played_on"]
    return steps_df


def do_steps_to_completion_plot(a, b, steps_df, label=""):
    if "game_lengths" in steps_df.columns:
        steps_df = steps_df[(steps_df.train == a) & (steps_df.play == b)]["game_lengths"]
        x_key = None
    else:
        x_key = (a, b)
    sb.kdeplot(x=x_key, data=steps_df, bw_adjust=1.5, cut=0, label=label)
    plt.gca().set_xlabel("steps to completion")
    plt.legend()
    plt.tight_layout()


def do_on_task_plot(steps_df):
    plt.figure(figsize=(5, 3))
    label = lambda a: f"{pretty_names[a]}"
    do_steps_to_completion_plot(0, 0, steps_df, label=label(0))
    do_steps_to_completion_plot(3, 3, steps_df, label=label(3))


def do_off_task_plot(steps_df):
    label = lambda a, b: f"off task: {pretty_names[a]} on {pretty_names[b]}"
    plt.figure(figsize=(5, 3))
    do_steps_to_completion_plot(0, 3, steps_df, label=label(0, 3))
    do_steps_to_completion_plot(3, 0, steps_df, label=label(3, 0))


def plot_k_panel_game(env, net, k, shared_colormap) -> Figure:
    assert len(env.states) == k
    states = env.states + [env.state]
    boards = [x.T @ [1, 2, 3, 4] for x in states]
    qs = [get_q(net, state) for state in states]
    qmin = min([q.min() for q in qs])
    qmax = max([np.ma.masked_invalid(q).max() for q in qs])
    rows = 2 * (k + 1) // 3
    f, ax = plt.subplots(
        rows, 3, figsize=(6, rows * 3), subplot_kw=dict(xticks=[], yticks=[])
    )
    ax_set1 = ax[: (rows // 2), :]
    ax_set2 = ax[(rows // 2) :, :]

    for board, subax in zip(boards, ax_set1.flatten().squeeze()):
        subax.imshow(board, cmap=game_cm, norm=norm)
    if shared_colormap:
        for q, subax in zip(qs, ax_set2.flatten().squeeze()):
            im = subax.imshow(q.T, cmap=q_cm, vmin=qmin, vmax=qmax)
        f.colorbar(im, ax=ax_set2, location="bottom", label="$-Q$")
    else:
        for q, subax in zip(qs, ax_set2.flatten().squeeze()):
            im = subax.imshow(q.T, cmap=q_cm)
            f.colorbar(im, ax=subax, location="bottom", label="$-Q$")
    return f
