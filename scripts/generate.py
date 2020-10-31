import argparse
import numpy as np
from blue_zero.config import Status
from path import Path


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


def gen_random(n, p_min, p_max,
               num_boards=1):
    boards = []
    for _ in range(num_boards):
        b = np.empty((n, n), dtype=np.int)
        b.fill(Status.wall)
        p = np.random.uniform(p_min, p_max)
        b[np.random.uniform(size=(n, n)) < p] = Status.alive
        boards.append(b)

    return np.stack(boards)


def main():
    args = parser.parse_args()
    try:
        p_min, p_max = args.p_range
    except TypeError:
        p_min, p_max = args.p, args.p
    boards = gen_random(args.n, p_min, p_max, num_boards=args.num_boards)
    if args.out_file.exists():
        raise FileExistsError(f"Output file {args.out_file} already exists. " 
                              "Aborting.")
    np.save(args.out_file, boards)


if __name__ == "__main__":
    main()
