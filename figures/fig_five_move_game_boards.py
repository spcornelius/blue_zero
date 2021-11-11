from blue_zero.agent import QAgent
from blue_zero.gui import BlueGUI
from blue_zero.mode import mode_registry
from blue_zero.mode.base import BlueMode

import helpers


if __name__ == "__main__":

    n = 10
    p = 0.8
    mode = 3

    net = helpers.get_net(trained_mode=mode, n=n)
    agent=helpers.QAgent(net)
    n_games = 80
    envs = [helpers.mode_registry[mode].from_random((n, n), p) for _ in range(n_games)]
    agent.play(envs)

    five_movers = [e for e in envs if e.steps_taken == 5]
    for idx,env in enumerate(five_movers):
        f = helpers.plot_six_panel_game(env)
        f.savefig(f"gameplay_n{n}_p{p}_mode{mode}.{idx}.png")