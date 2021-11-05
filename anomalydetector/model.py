from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation


class AdModel:
    @staticmethod
    def get_model(input_dim):
        """
        from DCASE2020_task2_baseline's model
        Copyright (c) 2020 Hitachi, Ltd.
        (MIT License)

        ref:
        https://github.com/y-kawagu/dcase2020_task2_baseline/
        define the keras model
        the model based on the simple dense auto encoder
        (128*128*128*128*8*128*128*128*128)
        """
        inputLayer = Input(shape=(input_dim,))

        h = Dense(128)(inputLayer)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(8)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(input_dim)(h)

        return Model(inputs=inputLayer, outputs=h)
