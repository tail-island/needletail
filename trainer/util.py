import numpy      as np
import tensorflow as tf

from keras import backend as K


def save_pb(model, path):
    K.set_learning_phase(False)

    outputs = [output.name.replace(':0', '') for output in model.outputs]
    # print(outputs)

    tf.train.write_graph(tf.graph_util.convert_variables_to_constants(K.get_session(), K.get_session().graph_def, outputs), '.', path, as_text=False)

    K.set_learning_phase(True)
