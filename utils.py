import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import logging
import sys
import numpy as np


def set_soft_gpu(soft_gpu):
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def save_gan(model, path):
    global z1, z2
    if "z1" not in globals():
        z1 = np.random.normal(0, 1, size=(9, 1, model.latent_dim))
    if "z2" not in globals():
        z2 = np.random.normal(0, 1, size=(9, 1, model.latent_dim))
    inputs = [np.ones((len(z1)*9, 1)), np.concatenate(
        (z1.repeat(9, axis=0).repeat(2, axis=1),
         np.repeat(np.concatenate([z2 for _ in range(9)], axis=0), model.n_style_block - 2, axis=1)),
        axis=1),
              np.zeros([len(z1) * 9, model.img_shape[0], model.img_shape[1]], dtype=np.float32)]
    z1_inputs = [np.ones((len(z1), 1)), z1.repeat(model.n_style_block, axis=1), np.zeros([len(z1), model.img_shape[0], model.img_shape[1]], dtype=np.float32)]
    z2_inputs = [np.ones((len(z2), 1)), z2.repeat(model.n_style_block, axis=1), np.zeros([len(z2), model.img_shape[0], model.img_shape[1]], dtype=np.float32)]

    imgs = model.predict(inputs)
    z1_imgs = model.predict(z1_inputs)
    z2_imgs = model.predict(z2_inputs)
    imgs = np.concatenate([z2_imgs, imgs], axis=0)
    rest_imgs = np.concatenate([np.ones([1, 128, 128, 3], dtype=np.float32), z1_imgs], axis=0)
    for i in range(len(rest_imgs)):
        imgs = np.concatenate([imgs[:i * 10], rest_imgs[i:i + 1], imgs[i * 10:]], axis=0)
    imgs = (imgs + 1) / 2

    plt.clf()
    nc, nr = 10, 10
    plt.figure(0, (nc*2, nr*2))
    for c in range(nc):
        for r in range(nr):
            i = r * nc + c
            plt.subplot(nr, nc, i + 1)
            plt.imshow(imgs[i])
            plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


def get_logger(date_str):
    log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = "visual/{}/train.log".format(date_str)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(log_fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger


class InstanceNormalization(keras.layers.Layer):
    def __init__(self, exclude_mean=True, trainable=None, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.trainable = trainable
        self.exclude_mean = exclude_mean
        self.beta, self.gamma = None, None

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=[1, 1, input_shape[-1]],
            initializer='ones',
            trainable=self.trainable)

        self.beta = self.add_weight(
            name='beta',
            shape=[1, 1, input_shape[-1]],
            initializer='zeros',
            trainable=self.trainable)

    def call(self, x, trainable=None):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        top = x if self.exclude_mean else (x - ins_mean)
        x_ins = top * (tf.math.rsqrt(ins_sigma + self.epsilon))
        out = x_ins * self.gamma + self.beta
        return out