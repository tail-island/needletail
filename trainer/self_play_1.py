import os
import pickle

from datetime       import datetime
from funcy          import *
from game           import *
from greedy_player  import *


def one_hot(action):
    return np.identity(32)[action]


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

    for i in range(100):
        game = random_game()

        state_1 = greedy_play(game, 0.2)
        state_2 = greedy_play(game, 0.2)

        write_play_data((game.points, state_1.actions, tuple(map(one_hot, rest(state_1.actions))) + ((0,) * 32,), value(state_1.distance, state_2.distance)))
        write_play_data((game.points, state_2.actions, tuple(map(one_hot, rest(state_2.actions))) + ((0,) * 32,), value(state_2.distance, state_1.distance)))

        print('{}\t{:0.3f}\t{:0.3f}'.format(i + 1, state_1.distance / greedy_play(game, 0).distance, state_2.distance / greedy_play(game, 0).distance))


if __name__ == '__main__':
    main()
