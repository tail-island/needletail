import cv2
import numpy      as np
import tensorflow as tf

from funcy    import *
from game     import *
from math     import sqrt
from operator import attrgetter, itemgetter


def import_graph_def(protocol_buffer_path):
    graph_def = tf.GraphDef()

    with protocol_buffer_path.open('rb') as f:
        graph_def.ParseFromString(f.read())

    tf.reset_default_graph()
    tf.import_graph_def(graph_def, name='')


def to_x(state):
    def blank_channel():
        return np.zeros((128, 128))

    def point_channel(action):
        result = blank_channel()

        result.itemset(state.game.points[action], 1)

        return result

    def actions_channel():
        result = blank_channel()

        for a1, a2 in zip(state.actions, rest(state.actions)):
            cv2.line(result, tuple(reversed(state.game.points[a1])), tuple(reversed(state.game.points[a2])), 1)

        return result

    return np.array((actions_channel(), point_channel(last(state.actions)), point_channel(first(state.actions))) +
                    tuple(map(lambda a: point_channel(a) if a in state.legal_actions else blank_channel(), range(1, 32)))).transpose(1, 2, 0)


def predict(state):
    def normalize_p(ps):
        ps = ps[list(state.legal_actions)]

        if len(ps) > 0 and min(ps) <= 0.01:
            ps = ps - min(ps) + 0.01

        ps = ps / (sum(ps) or 1)

        return ps

    x_tensor = tf.get_default_graph().get_tensor_by_name('input_1:0')
    p_tensor = tf.get_default_graph().get_tensor_by_name('activation_152/Softmax:0')
    v_tensor = tf.get_default_graph().get_tensor_by_name('activation_153/Tanh:0')

    x = to_x(state).reshape(1, 128, 128, 34)
    y = tf.get_default_session().run([p_tensor, v_tensor], feed_dict={x_tensor: x})

    return normalize_p(y[0][0]), y[1][0][0]


def pv_mcts_scores(state, evaluate_count):
    class Node:
        def __init__(self, state, p):
            self.state       = state
            self.p           = p
            self.w           = 0
            self.n           = 0
            self.child_nodes = None

        def evaluate(self):
            if self.state.is_end:
                v = predict(self.state)[1] if self.n == 0 else self.w / self.n

                self.w += v
                self.n += 1

                return v

            if not self.child_nodes:
                ps, v = predict(self.state)

                self.w += v
                self.n += 1

                self.child_nodes = tuple(map(lambda action, p: Node(self.state.next(action), p), self.state.legal_actions, ps))

                return v
            else:
                v = self.next_child_node().evaluate()

                self.w += v
                self.n += 1

                return v

        def next_child_node(self):
            def pucb_values():
                t = sum(map(attrgetter('n'), self.child_nodes))

                return tuple((child_node.w / child_node.n if child_node.n else 0) + 1.0 * child_node.p * sqrt(t) / (1 + child_node.n) for child_node in self.child_nodes)

            return self.child_nodes[np.argmax(pucb_values())]

    root_node = Node(state, 0)

    for _ in range(evaluate_count):
        root_node.evaluate()

    return tuple(map(attrgetter('n'), root_node.child_nodes))


def boltzman(xs, temperature):
    if temperature == 0:
        return np.identity(len(xs))[np.argmax(xs)]

    xs = [x ** (1 / temperature) for x in xs]

    return [x / sum(xs) for x in xs]


# def pv_mcts_play(game, evaluate_count, temperature):
#     def pv_mcts(state, ps_collection):
#         if state.is_end:
#             return state, ps_collection

#         scores = pv_mcts_scores(state, evaluate_count)
#         scores = np.array(scores) / sum(scores)

#         ps = np.zeros(32)
#         ps[list(state.legal_actions)] = scores

#         return pv_mcts(state.next(np.random.choice(state.legal_actions, p=boltzman(scores, temperature))), ps_collection + (ps,))

#     return pv_mcts(State(game), ())


def pv_mcts_play(game, protocol_buffer_path, evaluate_count, temperature):
    def pv_mcts(state, ps_collection):
        if state.is_end:
            return state, ps_collection

        scores = pv_mcts_scores(state, evaluate_count)
        scores = np.array(scores) / sum(scores)

        ps = np.zeros(32)
        ps[list(state.legal_actions)] = scores

        return pv_mcts(state.next(np.random.choice(state.legal_actions, p=boltzman(scores, temperature))), ps_collection + (ps,))

    import_graph_def(protocol_buffer_path)

    with tf.Session() as session:
        session.as_default()

        return pv_mcts(State(game), ())
