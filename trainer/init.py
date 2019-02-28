from functools          import reduce
from funcy              import *
from keras.layers       import *
from keras.models       import Model, save_model
from keras.regularizers import l2
from util               import *


def computational_graph():
    def ljuxt(*fs):
        return rcompose(juxt(*fs), list)

    def add():
        return Add()

    def batch_normalization():
        return BatchNormalization()

    def conv(width, kernel_size):
        return Conv2D(width, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

    def dense(width):
        return Dense(width, kernel_regularizer=l2(0.0005))

    def global_average_pooling():
        return GlobalAveragePooling2D()

    def max_pooling():
        return MaxPooling2D()

    def relu():
        return Activation('relu')

    def softmax():
        return Activation('softmax')

    def tanh():
        return Activation('tanh')

    ####

    def residual_net():
        def residual_unit(width, bottleneck_width):
            return rcompose(ljuxt(rcompose(batch_normalization(),
                                           relu(),
                                           conv(bottleneck_width, 1),
                                           batch_normalization(),
                                           relu(),
                                           conv(bottleneck_width, 3),
                                           batch_normalization(),
                                           relu(),
                                           conv(width, 1)),
                                  identity),
                            add())

        def residual_block(height, width, bottleneck_width):
            return rcompose(batch_normalization(),
                            relu(),
                            ljuxt(rcompose(conv(bottleneck_width, 1),
                                           batch_normalization(),
                                           relu(),
                                           conv(bottleneck_width, 3),
                                           batch_normalization(),
                                           relu(),
                                           conv(width, 1)),
                                  conv(width, 1)),
                            add(),
                            rcompose(*repeatedly(partial(residual_unit, width, bottleneck_width), height - 1)))

        return rcompose(conv(64, 3),
                        max_pooling(),
                        residual_block( 3,  256,  64),
                        max_pooling(),
                        residual_block( 8,  512, 128),
                        max_pooling(),
                        residual_block(36, 1024, 256),
                        max_pooling(),
                        residual_block( 3, 2048, 512),
                        batch_normalization(),
                        relu(),
                        global_average_pooling())

    return rcompose(residual_net(),
                    ljuxt(rcompose(dense(32), softmax()),
                          rcompose(dense( 1), tanh())))


def main():
    model = Model(*juxt(identity, computational_graph())(Input(shape=(128, 128, 34))))
    model.summary()

    save_pb(model, './data/model/0000.pb')
    save_pb(model, './data/model/candidate/0000.pb')

    save_model(model, './data/model/candidate/0000.h5')


if __name__ == '__main__':
    main()
