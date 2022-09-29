from collections import Counter

import play, helpers

if __name__ == "__main__":

    n = 20
    p = 0.55
    mode = 4
    n_games = 600
    models_root = "/app/"
    game_len = 5
    shared_colormap = False

    envs = play.get_played_envs(
        trained_mode=mode,
        game_mode=mode,
        n=n,
        p=p,
        n_games=n_games,
        models_root=models_root,
    )
    games = [e for e in envs if e.steps_taken == game_len]
    print(sorted(Counter([e.steps_taken for e in envs]).items()))
    net = play.get_net(trained_mode=mode, n=n, models_root=models_root)
    for idx, env in enumerate(games):
        f = helpers.plot_k_panel_game(env, net, game_len, shared_colormap, cols=3)
        f.savefig(f"gameplay_n{n}_p{p}_mode{mode}.{idx}.png")
        helpers.plt.close()
