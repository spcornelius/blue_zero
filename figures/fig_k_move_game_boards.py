import helpers
from collections import Counter

if __name__ == "__main__":

    n = 10
    p = 0.8
    mode = 0
    n_games = 200
    models_root = "/app/"
    game_len = 2
    
    envs = helpers.get_played_envs(
        trained_mode=mode,
        game_mode=mode,
        n=n,
        p=p,
        n_games=n_games,
        models_root=models_root,
    )
    games = [e for e in envs if e.steps_taken == game_len]
    print(sorted(Counter([e.steps_taken for e in envs]).items()))
    net = helpers.get_net(trained_mode=mode, n=n, models_root=models_root)
    for idx, env in enumerate(games):
        f = helpers.plot_k_panel_game(env, net, game_len)
        f.savefig(f"gameplay_n{n}_p{p}_mode{mode}.{idx}.png")
