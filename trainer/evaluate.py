import os

from funcy          import *
from game           import *
from greedy_player  import *
from pathlib        import Path
from pv_mcts_player import *
from shutil         import copyfile


def write_protocol_buffer(challenger_path):
    copyfile(challenger_path, Path('./data/model/').joinpath(challenger_path.name))


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    challenger_path = last(sorted(Path('./data/model/candidate').glob('*.pb')))
    champion_path   = last(sorted(Path('./data/model').glob('*.pb')))

    c = 0

    for i in range(40):
        game = random_game()

        state_1, _ = pv_mcts_play(game, challenger_path, 50, 0.0)
        state_2, _ = pv_mcts_play(game, champion_path,   50, 0.0)

        if state_1.distance <= state_2.distance:
            c += 1

        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(i + 1, c / (i + 1), state_1.distance / greedy_play(game, 0.0).distance, state_2.distance / greedy_play(game, 0.0).distance))

    if c / 40 >= 0.55:
        write_protocol_buffer(challenger_path)


if __name__ == '__main__':
    main()
