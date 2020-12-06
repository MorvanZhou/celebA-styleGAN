import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import *
from utils import InstanceNormalization


class AdaNorm(keras.layers.Layer):
    def __init__(self, exclude_mean=True, epsilon=1e-5):
        super().__init__()
        self.exclude_mean = exclude_mean
        self.epsilon = epsilon

    def call(self, x, trainable=None):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        top = x if self.exclude_mean else x - ins_mean
        x = top * (tf.math.rsqrt(ins_sigma + self.epsilon))
        return x


class AdaMod(keras.layers.Layer):
    def __init__(self, trainable=None):
        super().__init__()
        self.trainable = trainable
        self.ys, self.yb = None, None

    def build(self, input_shape):
        x_input_shape, w_input_shape = input_shape
        self.ys = keras.Sequential([
            keras.layers.Dense(x_input_shape[-1], input_shape=w_input_shape[1:], trainable=self.trainable),
            keras.layers.Reshape([1, 1, -1])
        ])
        self.yb = keras.Sequential([
            keras.layers.Dense(x_input_shape[-1], input_shape=w_input_shape[1:], trainable=self.trainable),
            keras.layers.Reshape([1, 1, -1])
        ])  # [1, 1, c] per feature map

    def call(self, inputs, training=None):
        x, w = inputs
        o = self.ys(w, training=training) * x + self.yb(w, training=training)
        return o


class StyleGAN(keras.Model):
    def __init__(self, img_shape, latent_dim, exclude_mean,
                 summary_writer=None, lr=0.0002, beta1=0.5, beta2=0.99, lambda_=10, ls_loss=False):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.ls_loss = ls_loss
        self.exclude_mean = exclude_mean
        self._b_scale_count = 0
        self.lambda_ = lambda_

        self.const = tf.random.normal([4, 4, 128], 0, 0.05)
        self.f = self._get_f()
        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)
        self.d_loss_fun = keras.losses.MeanSquaredError() if self.ls_loss \
            else keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

        self.summary_writer = summary_writer
        self._train_step = 0

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs[0], np.ndarray):
            inputs = (tf.convert_to_tensor(i) for i in inputs)
        return self.g.call(inputs, training=training)

    @staticmethod
    def _get_f():
        f = keras.Sequential([
            Dense(128),
            LeakyReLU(0.2),
            Dense(128),
            LeakyReLU(0.2),
            Dense(128),
            LeakyReLU(0.2),
        ], name="f")
        return f

    def _get_generator(self):
        z1 = keras.Input((self.latent_dim,))
        z2 = keras.Input((self.latent_dim,))
        noise_ = keras.Input((self.img_shape[0], self.img_shape[1]))
        noise = tf.expand_dims(noise_, axis=-1)

        w1 = self.f(z1)
        w2 = self.f(z2)

        x = self.add_noise(self.const, noise)
        x = AdaNorm(exclude_mean=self.exclude_mean)(x)
        x = self.style_block(128, x, w1, noise, upsampling=False)  # 4^2
        x = self.style_block(128, x, w1, noise)  # 8^2
        x = self.style_block(128, x, w1, noise)  # 16^2
        x = self.style_block(128, x, w1, noise)  # 32^2
        x = self.style_block(128, x, w2, noise)  # 64^2
        x = self.style_block(128, x, w2, noise)  # 128^2
        o = Conv2D(3, 7, 1, "same", activation="tanh")(x)
        g = keras.Model([z1, z2, noise_], o, name="generator")
        g.summary()
        return g

    def style_block(self, filters, x, w, b_noise, upsampling=True):
        x = AdaMod()((x, w))
        if upsampling:
            x = keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)

        x = keras.layers.Conv2D(filters, 3, 1, "same")(x)
        x = self.add_noise(x, b_noise)
        x = keras.layers.ReLU()(x)
        x = AdaNorm(exclude_mean=self.exclude_mean)(x)
        return x

    def add_noise(self, x, b_noise):
        x_shape = x.shape[1:] if x.shape[0] is None else x.shape
        b_noise_ = b_noise[:, :x_shape[0], :x_shape[1], :]
        scale = self.add_weight(name="b_scale{}".format(self._b_scale_count), shape=[1, 1, x.shape[-1]])
        self._b_scale_count += 1
        return scale * b_noise_ + x

    def _get_discriminator(self):
        def add_block(filters, do_norm=True):
            model.add(Conv2D(filters, 3, strides=2, padding='same',
                             kernel_initializer=keras.initializers.RandomNormal(0.02)))
            if do_norm: model.add(InstanceNormalization(exclude_mean=exclude_mean))
            model.add(LeakyReLU(alpha=0.2))

        exclude_mean = self.exclude_mean
        model = keras.Sequential([Input(self.img_shape)], name="d")
        # [n, 128, 128, 3]
        # model.add(GaussianNoise(0.02))
        add_block(64, do_norm=False)   # -> 64^2
        add_block(128)                   # -> 32^2
        add_block(256)                  # -> 16^2
        add_block(256)                  # -> 8^2
        # add_block(512)                # -> 4^2

        model.add(Conv2D(128, 3, 2, "valid"))
        model.add(Flatten())
        model.add(keras.layers.Dense(1))

        model.summary()
        return model

    # gradient penalty
    def gp(self, real_img, fake_img):
        e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
        noise_img = e * real_img + (1. - e) * fake_img  # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            o = self.d(noise_img)
        g = tape.gradient(o, noise_img)  # image gradients
        g_norm2 = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
        gp = tf.square(g_norm2 - 1.)
        return tf.reduce_mean(gp)

    @staticmethod
    def w_distance(real, fake):
        # the distance of two data distributions
        return tf.reduce_mean(real) - tf.reduce_mean(fake)

    def train_d(self, img):
        n = len(img)
        z1 = tf.random.normal((n, self.latent_dim))
        z2 = tf.random.normal((n, self.latent_dim)) if np.random.random() < 0.5 else z1
        noise = tf.random.normal((n, self.img_shape[0], self.img_shape[1]))
        inputs = (z1, z2, noise)
        with tf.GradientTape() as tape:
            gimg = self.call(inputs, training=False)
            gp = self.gp(img, gimg)
            pred_fake = self.d.call(gimg, training=True)
            pred_real = self.d.call(img, training=True)
            w_distance = -self.w_distance(pred_real, pred_fake)  # maximize W distance
            gp_loss = self.lambda_ * gp
            loss = gp_loss + w_distance
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d/w_distance", w_distance, step=self._train_step)
                tf.summary.scalar("d/gp", gp, step=self._train_step)
        return gp, w_distance

    def train_g(self, n):
        z1 = tf.random.normal((n, self.latent_dim))
        z2 = tf.random.normal((n, self.latent_dim)) if np.random.random() < 0.5 else z1
        noise = tf.random.normal((n, self.img_shape[0], self.img_shape[1]))
        inputs = (z1, z2, noise)
        with tf.GradientTape() as tape:
            gimg = self.call(inputs, training=True)
            pred_fake = self.d.call(gimg, training=False)
            w_distance = tf.reduce_mean(-pred_fake)  # minimize W distance
        var = self.g.trainable_variables + self.f.trainable_variables
        grads = tape.gradient(w_distance, var)
        self.opt.apply_gradients(zip(grads, var))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g/w_distance", w_distance, step=self._train_step)
                if self._train_step % 1000 == 0:
                    tf.summary.image("gimg", (gimg + 1) / 2, max_outputs=5, step=self._train_step)

        return w_distance

    def step(self, img):
        gw = self.train_g(len(img) * 2)
        dgp, dw = self.train_d(img)
        self._train_step += 1
        return gw, dgp, dw