import numpy as np

from funcy     import *
from itertools import product
from operator  import contains, getitem
from random    import sample


def distance(point_1, point_2):
    return np.linalg.norm(np.array(point_1) - np.array(point_2))


class Game:
    def __init__(self, points):
        self.points = points


def random_game():
    return Game(sample(tuple(product(range(1, 127), range(1, 127))), 8))


class State:
    def __init__(self, game, actions=(0,)):
        self.game    = game;
        self.actions = actions

    @property
    def legal_actions(self):
        return tuple(filter(complement(partial(contains, self.actions)), range(len(self.game.points))))

    @property
    def is_end(self):
        return len(self.actions) == len(self.game.points)

    @property
    def distance(self):
        return sum(map(lambda a1, a2: distance(self.game.points[a1], self.game.points[a2]), self.actions, rest(self.actions + (0,))))

    def next(self, action):
        return State(self.game, self.actions + (action,))
