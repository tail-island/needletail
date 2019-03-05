import os
import random
import tensorflow as tf

from funcy          import *
from game           import *
from greedy_player  import *
from pathlib        import Path
from pv_mcts_player import *
from statistics     import mean, median


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    for model_path in sorted(Path('./data/model').glob('*.pb')):
        import_graph_def(model_path)

        with tf.Session() as session:
            random.seed(0)

            distances = []
            win_count = 0

            for _ in range(100):
                game = random_game()

                distance_1 = pv_mcts_play(game, 50, 0.0)[0].distance
                distance_2 = greedy_play(game, 0.0).distance

                distances.append(distance_1 / distance_2)
                win_count += distance_1 <= distance_2

            print('{}\t{}\t{}\t{}'.format(mean(distances), median(distances), win_count / 100, model_path.name))


if __name__ == '__main__':
    main()
