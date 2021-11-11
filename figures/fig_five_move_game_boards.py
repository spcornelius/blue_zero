import helpers

if __name__ == "__main__":

    n = 10
    p = 0.8
    mode = 3
    n_games = 80
    models_root = "/app/"

    envs = helpers.get_played_envs(
        trained_mode=mode,
        game_mode=mode,
        n=n,
        p=p,
        n_games=n_games,
        models_root=models_root,
    )
    five_movers = [e for e in envs if e.steps_taken == 5]

    for idx, env in enumerate(five_movers):
        f = helpers.plot_six_panel_game(env)
        f.savefig(f"gameplay_n{n}_p{p}_mode{mode}.{idx}.png")
