from glob import glob
from itertools import product as iterproduct
from pathlib import Path

import pandas as pd

from blue_zero.agent import EpsGreedyQAgent, QAgent
from blue_zero.mode import mode_registry
from blue_zero.mode.base import BlueMode
from blue_zero.qnet import QNet


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


def get_played_envs(trained_mode, game_mode, n, p, n_games, models_root, random=False):
    net = get_net(trained_mode=trained_mode, n=n, models_root=models_root)
    envs = [get_env(game_mode, n, p) for _ in range(n_games)]
    if random:
        EpsGreedyQAgent(net, eps=1).play(envs, pbar=True)
    else:
        QAgent(net).play(envs, pbar=True)
    return envs


def make_game_length_filename(trained_mode, game_mode, n, p):
    return f"train_{trained_mode}_play_{game_mode}_n_{n}_p_{p:.4f}"


def parse_game_length_filename(filename):
    keys = filename.split("_")[::2]
    vals = filename.split("_")[1::2]
    vals = [float(i) if "." in i else int(i) for i in vals]
    return dict(zip(keys, vals))


def generate_game_lengths(
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
    fname = Path(output_root) / make_game_length_filename(trained_mode, game_mode, n, p)
    record_game_lengths(fname, envs)


def record_game_lengths(filename, envs):
    with open(filename, "a") as f:
        f.write("\n".join([str(e.steps_taken) for e in envs]))
        f.write("\n")


def load_game_lengths(output_root):
    runs = []
    for fname in glob(str(Path(output_root) / "*")):
        with open(fname) as f:
            game_lengths = [int(i) for i in f.readlines()]
        run_data = parse_game_length_filename(fname.split("/")[-1])
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
