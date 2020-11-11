import argparse
import numpy as np
from path import Path
from blue_zero.config import Status
from blue_zero.env.util import env_cls


parser = argparse.ArgumentParser()
parser.add_argument('--num-boards', type=int, required=True,
                    help="number of boards to generate")
parser.add_argument('-n', type=int, required=True,
                    help="size of board (side length)")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-p', type=float,
                   help="green probability")
group.add_argument('--p-range', metavar=('P_MIN', 'P_MAX'),
                   type=float, nargs=2,
                   help="range of green probabilities (uniform sampling)")

parser.add_argument('-o', dest='out_file', metavar='OUT_FILE',
                    type=Path, required=True,
                    help="where to save generated boards")
parser.add_argument('--mode', dest='mode',
                    type=int, required=True,
                    help="game mode")
parser.add_argument('--direction', type=str, default='horizontal',
                    choices=['horizontal', 'vertical', 'both'],
                    help="current direction (ignored unless mode = 3)")


def gen_random(n: int, p_min: float, p_max: float, mode: int,
               num_boards=1, **kwargs):
    boards = []
    while len(boards) < num_boards:
        b = np.empty((n, n), dtype=np.int)
        b.fill(Status.wall)
        p = np.random.uniform(p_min, p_max)
        b[np.random.uniform(size=(n, n)) < p] = Status.alive

        # only retain states that aren't already terminal
        env = env_cls[mode](b, **kwargs)
        if env.done:
            continue
        boards.append(b)

    return np.stack(boards)


def main():
    args = vars(parser.parse_args())
    try:
        p_min, p_max = args.pop('p_range')
    except TypeError:
        p = args.pop('p')
        p_min, p_max = p, p

    n = args.pop('n')
    mode = args.pop('mode')
    out_file = args.pop('out_file')
    num_boards = args.pop('num_boards')
    boards = gen_random(n, p_min, p_max, mode, num_boards=num_boards,
                        **args)
    if out_file.exists():
        raise FileExistsError(f"Output file {args.out_file} already exists. "
                              "Aborting.")
    np.save(out_file, boards)


if __name__ == "__main__":
    main()
