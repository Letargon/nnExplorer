import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Sequential
from tensorboard_master import HP_CONV_CORE, HP_LR, HP_OPTIMIZER, HP_IMAGE_SIZE


class ModelGenerator:
    @staticmethod
    def base_cnn(hparams):
        # create model
        input_shape = (hparams[HP_IMAGE_SIZE], hparams[HP_IMAGE_SIZE], 3)
        model = Sequential()
        model.add(Conv2D(8, kernel_size=hparams[HP_CONV_CORE], activation='relu', input_shape=input_shape))
        model.add(Conv2D(16, kernel_size=hparams[HP_CONV_CORE], activation='relu'))
        model.add(Conv2D(32, kernel_size=hparams[HP_CONV_CORE], activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        if hparams[HP_OPTIMIZER] == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=hparams[HP_LR])

        if hparams[HP_OPTIMIZER] == "sgd":
            optimizer = tf.keras.optimizers.SGD(lr=hparams[HP_LR])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def get_shape(image_size, color_mode="grayscale"):
        if color_mode == "grayscale":
            return image_size[0], image_size[1], 1
        if color_mode == "rgb":
            return image_size[0], image_size[1], 3

    @staticmethod
    def base_cnn_maxpool(hparams):
        # create model
        input_shape = (hparams[HP_IMAGE_SIZE], hparams[HP_IMAGE_SIZE], 1)
        model = Sequential()
        model.add(Conv2D(8, kernel_size=hparams[HP_CONV_CORE], activation='relu', input_shape=input_shape))
        model.add(MaxPool2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(16, kernel_size=hparams[HP_CONV_CORE], activation='relu'))
        model.add(Conv2D(32, kernel_size=hparams[HP_CONV_CORE], activation='relu'))
        model.add(MaxPool2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        if hparams[HP_OPTIMIZER] == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=hparams[HP_LR])

        if hparams[HP_OPTIMIZER] == "sgd":
            optimizer = tf.keras.optimizers.SGD(lr=hparams[HP_LR])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def base_cnn_maxpool_batch_normalization(hparams):
        # create model
        input_shape = (hparams[HP_IMAGE_SIZE], hparams[HP_IMAGE_SIZE], 3)
        model = Sequential()
        model.add(Conv2D(8, kernel_size=hparams[HP_CONV_CORE], input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPool2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(16, kernel_size=hparams[HP_CONV_CORE]))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(32, kernel_size=hparams[HP_CONV_CORE]))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPool2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        if hparams[HP_OPTIMIZER] == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=hparams[HP_LR])

        if hparams[HP_OPTIMIZER] == "sgd":
            optimizer = tf.keras.optimizers.SGD(lr=hparams[HP_LR])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def base_dense(hparams):
        # create model
        input_shape = (hparams[HP_IMAGE_SIZE], hparams[HP_IMAGE_SIZE], 1)
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        if hparams[HP_OPTIMIZER] == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=hparams[HP_LR])

        if hparams[HP_OPTIMIZER] == "sgd":
            optimizer = tf.keras.optimizers.SGD(lr=hparams[HP_LR])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def base_dense_normalized(hparams):
        # create model
        input_shape = (hparams[HP_IMAGE_SIZE], hparams[HP_IMAGE_SIZE], 1)
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(8, activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(8, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        if hparams[HP_OPTIMIZER] == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=hparams[HP_LR])

        if hparams[HP_OPTIMIZER] == "sgd":
            optimizer = tf.keras.optimizers.SGD(lr=hparams[HP_LR])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def littleMobileNet(hparams):
        input_shape = (hparams[HP_IMAGE_SIZE], hparams[HP_IMAGE_SIZE], 1)
        model = Sequential()

        model.add(Conv2D(8, kernel_size=hparams[HP_CONV_CORE], input_shape=input_shape))

        model.add(DepthwiseConv2D(kernel_size=hparams[HP_CONV_CORE]))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(8, kernel_size=(1, 1)))

        model.add(DepthwiseConv2D(kernel_size=hparams[HP_CONV_CORE], strides=2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(16, kernel_size=(1, 1)))

        model.add(DepthwiseConv2D(kernel_size=hparams[HP_CONV_CORE], strides=1))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(32, kernel_size=(1, 1)))

        model.add(DepthwiseConv2D(kernel_size=hparams[HP_CONV_CORE], strides=2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(64, kernel_size=(1, 1)))

        model.add(Flatten())

        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))

        if hparams[HP_OPTIMIZER] == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=hparams[HP_LR])

        if hparams[HP_OPTIMIZER] == "sgd":
            optimizer = tf.keras.optimizers.SGD(lr=hparams[HP_LR])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
