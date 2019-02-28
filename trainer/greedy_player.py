import numpy as np

from funcy          import *
from game           import *
from pv_mcts_player import *
from operator       import getitem


def greedy_play(game, temperature):
    def greedy(state):
        if state.is_end:
            return state

        scores = np.array(tuple(map(partial(distance, game.points[last(state.actions)]), map(partial(getitem, game.points), state.legal_actions))))
        scores = scores / sum(scores)
        scores = max(scores) - scores + 0.01
        scores = scores / sum(scores)

        return greedy(state.next(np.random.choice(state.legal_actions, p=boltzman(scores, temperature))))

    return greedy(State(game))
