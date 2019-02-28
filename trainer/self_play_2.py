import numpy as np
import os
import pickle

from datetime       import datetime
from funcy          import *
from game           import *
from greedy_player  import *
from pathlib        import Path
from pv_mcts_player import *


def value(distance_1, distance_2):
    if distance_1 < distance_2:
        return  1

    if distance_1 > distance_2:
        return -1

    return 0


def write_play_data(play_data):
    now = datetime.now()
    data_file_path = './data/play/{:04}-{:02}-{:02}-{:02}-{:02}-{:02}-{:06}{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, os.uname()[1], os.getpid())

    with open('{}.pickle'.format(data_file_path), 'wb') as f:
        pickle.dump(play_data, f)


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    challenger_path = last(sorted(Path('./data/model/candidate').glob('*.pb')))
    champion_path   = last(sorted(Path('./data/model').glob('*.pb')))

    for i in range(100):
        game = random_game()

        state_1, ps_collection_1 = pv_mcts_play(game, challenger_path, 50, 0.2)
        state_2, ps_collection_2 = pv_mcts_play(game, champion_path,   50, 0.2)

        write_play_data((game.points, state_1.actions, tuple(ps_collection_1) + ((0,) * 32,), value(state_1.distance, state_2.distance)))
        write_play_data((game.points, state_2.actions, tuple(ps_collection_2) + ((0,) * 32,), value(state_2.distance, state_1.distance)))

        print('{}\t{:.3f}\t{:.3f}'.format(i + 1, state_1.distance / greedy_play(game, 0).distance, state_2.distance / greedy_play(game, 0).distance))


if __name__ == '__main__':
    main()
