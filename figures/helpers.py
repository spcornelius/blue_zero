import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib import colors
from matplotlib.figure import Figure

from blue_zero import config as cfg
import play


blue_colors = [
    cfg.black,
    cfg.green,
    cfg.blue,
    cfg.orange,
]
pretty_names = {-1: "random", 0: "network", 3: "flow", 4: "noodle"}
blue_colors = np.array(blue_colors) / 255
game_cm = colors.ListedColormap(blue_colors)
norm = colors.BoundaryNorm(boundaries=[1, 2, 3, 4, 5], ncolors=game_cm.N)

q_cm = matplotlib.cm.get_cmap("plasma_r").copy()
q_cm.set_bad(color="black")


def plot_q_modes():
    net0 = play.get_net(0)
    net3 = play.get_net(3)
    net4 = play.get_net(4)
    env = play.get_env(p=0.65)
    q0 = play.get_q(net0, env.state)
    q3 = play.get_q(net3, env.state)
    q4 = play.get_q(net4, env.state)
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


def plot_steps_to_completion(train_mode, play_mode, steps_df, label=""):
    if "game_lengths" in steps_df.columns:
        steps_df = steps_df[
            (steps_df.train == train_mode) & (steps_df.play == play_mode)
        ]["game_lengths"]
        x_key = None
    else:
        x_key = (train_mode, play_mode)
    sb.kdeplot(x=x_key, data=steps_df, bw_adjust=1.5, cut=0, label=label)
    plt.gca().set_xlabel("steps to completion")
    plt.legend()
    plt.tight_layout()


def plot_on_task(steps_df):
    plt.figure(figsize=(6, 4))
    label = lambda a: f"{pretty_names[a]}"
    plot_steps_to_completion(0, 0, steps_df, label=label(0))
    plot_steps_to_completion(3, 3, steps_df, label=label(3))


def plot_off_task(steps_df):
    label = lambda a, b: f"off task: {pretty_names[a]} on {pretty_names[b]}"
    plt.figure(figsize=(6, 4))
    plot_steps_to_completion(0, 3, steps_df, label=label(0, 3))
    plot_steps_to_completion(3, 0, steps_df, label=label(3, 0))


def plot_k_panel_game(env, net, k, shared_colormap, cols=3) -> Figure:
    assert len(env.states) == k
    states = env.states + [env.state]
    boards = [x.T @ [1, 2, 3, 4] for x in states]
    qs = [play.get_q(net, state) for state in states]
    subplot_scale = 2
    rows = 2 * (k + 1) // cols
    figsize = (cols * subplot_scale, rows * subplot_scale)
    f, ax = plt.subplots(
        rows, cols, figsize=figsize, subplot_kw=dict(xticks=[], yticks=[])
    )
    top_rows_ax = ax[: (rows // 2), :].flatten().squeeze()
    bottom_rows_ax = ax[(rows // 2) :, :].flatten().squeeze()

    _plot_boards(boards, top_rows_ax)
    if shared_colormap:
        _plot_q_values_shared_colormap(qs, f, bottom_rows_ax)
    else:
        _plot_q_values_individual_colormaps(qs, f, bottom_rows_ax)
    return f


def _plot_q_values_individual_colormaps(qs, f, ax_set):
    for q, subax in zip(qs, ax_set):
        im = subax.imshow(q.T, cmap=q_cm)
        f.colorbar(im, ax=subax, location="bottom", label="$-Q$")


def _plot_q_values_shared_colormap(qs, f, ax_set):
    qmin = min([q.min() for q in qs])
    qmax = max([np.ma.masked_invalid(q).max() for q in qs])
    for q, subax in zip(qs, ax_set):
        im = subax.imshow(q.T, cmap=q_cm, vmin=qmin, vmax=qmax)
    f.colorbar(im, ax=ax_set, location="bottom", label="$-Q$")


def _plot_boards(boards, ax_set):
    for board, subax in zip(boards, ax_set):
        subax.imshow(board, cmap=game_cm, norm=norm)
