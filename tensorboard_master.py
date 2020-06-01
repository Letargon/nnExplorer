from tensorflow.keras import backend as K

import numpy as np
import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

METRIC_ACCURACY = "accuracy"

HP_CONV_CORE = hp.HParam('conv_core', hp.Discrete([3, 5]))

HP_LR = hp.HParam("learning rate", hp.Discrete([1e-3, 1e-4, 1e-5, 1e-6]))

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(["adam", "sgd"]))

HP_IMAGE_SIZE = hp.HParam('image_size', hp.Discrete([32, 64]))


def default_callback(log_dir):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def hyperparams_callback(log_dir, hparams):
    return hp.KerasCallback(log_dir, hparams)


def record_params(hparams):
    # hp.hparams_config(
    #     hparams,
    #     metrics=[hp.Metric("accuracy", display_name='Train_accuracy'), hp.Metric("val_accuracy", display_name="Val_accuracy")],
    # )
    hp.hparams(hparams)


def record_scalar(scalar, name):
    tf.summary.scalar(name, scalar, step=1)
