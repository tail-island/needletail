import numpy as np
import pickle

from funcy           import *
from keras.callbacks import LearningRateScheduler
from keras.models    import load_model, save_model
from pathlib         import Path
from pv_mcts_player  import *
from operator        import getitem
from random          import sample
from util            import *


def generator_func(batch_size):
    def load_data():
        def load_game(path):
            with path.open(mode='rb') as f:
                points, actions, ps, v = pickle.load(f)

            return map(lambda i: (State(Game(points), actions[:i + 1]), ps[i], v), range(len(actions)))

        candidates = tuple(mapcat(load_game, tuple(sorted(Path('./data/play').glob('*.pickle')))[-100 * 2 * 50:]))

        return tuple(sample(candidates, 2000 if len(candidates) >= 2000 else len(candidates)))

    def generator(data):
        for steps in partition(batch_size, cat(map(np.random.permutation, repeat(data)))):
            states, ps, vs = zip(*steps)

            yield np.array(tuple(map(to_x, states))), [np.array(ps), np.array(vs)]

    data = load_data()

    return generator(data), len(data) // batch_size


def main():
    generator, steps_per_epoch = generator_func(20)

    model_path = last(sorted(Path('./data/model/candidate').glob('*.h5')))
    model = load_model(str(model_path))

    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer='adam')
    model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch, epochs=50, callbacks=[LearningRateScheduler(partial(getitem, tuple(concat(repeat(0.001, 30), repeat(0.0005, 10), repeat(0.00025, 10)))))])

    save_pb(model, str(model_path.with_name('{:04}.pb'.format(int(model_path.stem) + 1))))
    save_model(model, str(model_path.with_name('{:04}.h5'.format(int(model_path.stem) + 1))))


if __name__ == '__main__':
    main()
