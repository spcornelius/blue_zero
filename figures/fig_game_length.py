import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import helpers, play

sb.set_theme("paper")


n = 20
pvec = []#[0.7, 0.75, 0.8, 0.85, 0.9]
path_to_game_lengths = "/app/figures/game_lengths/"
img_type="pdf"

if __name__ == "__main__":
    game_lengths = play.load_game_lengths(path_to_game_lengths)
    df = pd.concat(map(pd.DataFrame, game_lengths))
    df = df[df.n == n]
    for p in pvec:
        helpers.plot_on_task(df[df.p == p])
        plt.savefig(f"on_task_n{n}_p{p}.{img_type}")
        helpers.plot_off_task(df[df.p == p])
        plt.savefig(f"off_task_n{n}_p{p}.{img_type}")
    df.train = df.train.map(helpers.pretty_names)
    df.play = df.play.map(helpers.pretty_names)
    df = df.rename(columns={"train": "agent"})
    play_train_comparisons = [
        (["random", "flow", "network"], "network"),
        (["random", "flow", "network"], "flow"),
        (["random", "noodle", "network"], "network"),
        (["random", "noodle", "network"], "noodle"),
    ]
        
    for comparisons, mode in play_train_comparisons:
        plt.figure(figsize=(5, 3))
        sb.lineplot(
            x="p",
            y="game_lengths",
            hue="agent",
            hue_order=comparisons,
            ci="sd",
            palette="Set2",
            data=df[df.play == mode].reset_index(),
        )
        plt.ylabel("steps to completion")
        plt.tight_layout()
        plt.savefig(f"game_lengths_n{n}_{'+'.join(comparisons)}_playing_{mode}.{img_type}")
