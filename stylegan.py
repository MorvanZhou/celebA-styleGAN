import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import *
from utils import InstanceNormalization
import tensorflow.keras.initializers as initer


class AdaNorm(keras.layers.Layer):
    def __init__(self, exclude_mean=False, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.exclude_mean = exclude_mean

    def call(self, x, **kwargs):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        top = x if self.exclude_mean else x - ins_mean
        x_ins = top * (tf.math.rsqrt(ins_sigma + self.epsilon))
        return x_ins


class AdaMod(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.ys, self.yb = None, None

    def call(self, inputs, **kwargs):
        x, w = inputs
        s, b = self.ys(w), self.yb(w)
        o = s * x + b
        return o

    def build(self, input_shape):
        x_shape, w_shape = input_shape
        self.ys = keras.Sequential([
            keras.layers.Dense(x_shape[-1], input_shape=w_shape[1:], name="s",
                               kernel_initializer=initer.RandomNormal(0, 1),
                               bias_initializer=initer.Constant(1)
                               ),   # this kernel and bias is important
            keras.layers.Reshape([1, 1, -1])
        ])
        self.yb = keras.Sequential([
            keras.layers.Dense(x_shape[-1], input_shape=w_shape[1:], name="b",
                               kernel_initializer=initer.RandomNormal(0, 1)),
            keras.layers.Reshape([1, 1, -1])
        ])  # [1, 1, c] per feature map


class AddNoise(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.s = None
        self.x_shape = None

    def call(self, inputs, **kwargs):
        x, noise = inputs
        noise_ = noise[:, :self.x_shape[1], :self.x_shape[2], :]
        return self.s * noise_ + x

    def build(self, input_shape):
        self.x_shape, _ = input_shape
        self.s = self.add_weight(name="noise_scale", shape=[1, 1, self.x_shape[-1]],
                                 initializer=initer.RandomNormal(0., .5))   # large initial noise


class Map(keras.layers.Layer):
    def __init__(self, size, num_layers):
        super().__init__()
        self.size = size
        self.num_layers = num_layers
        self.f = None

    def call(self, inputs, **kwargs):
        w = self.f(inputs)
        return w

    def build(self, input_shape):
        self.f = keras.Sequential([
            Dense(self.size, input_shape=input_shape[1:]),
            # keras.layers.LeakyReLU(0.2),  # worse performance when using non-linearity in mapping
        ]+[Dense(self.size) for _ in range(self.num_layers - 1)]
        )


class Style(keras.layers.Layer):
    def __init__(self, filters, upsampling=True, exclude_mean=False):
        super().__init__()
        self.filters = filters
        self.upsampling = upsampling
        self.exclude_mean = exclude_mean
        self.ada_mod, self.ada_norm, self.add_noise, self.up, self.conv = None, None, None, None, None

    def call(self, inputs, **kwargs):
        x, w, noise = inputs
        x = self.ada_mod((x, w))
        if self.up is not None:
            x = self.up(x)
        x = self.conv(x)
        x = self.ada_norm(x)
        x = keras.layers.LeakyReLU()(x)
        x = self.add_noise((x, noise))
        return x

    def build(self, input_shape):
        self.ada_mod = AdaMod()
        self.ada_norm = AdaNorm(exclude_mean=self.exclude_mean)
        if self.upsampling:
            self.up = keras.layers.UpSampling2D((2, 2), interpolation="bilinear")
        self.add_noise = AddNoise()
        self.conv = keras.layers.Conv2D(self.filters, 3, 1, "same")


def get_generator(latent_dim, img_shape, exclude_mean):
    n_style_block = 0
    const_size = _size = 8
    while _size <= img_shape[1]:
        n_style_block += 1
        _size *= 2

    z = keras.Input((n_style_block, latent_dim,), name="z")
    noise_ = keras.Input((img_shape[0], img_shape[1]), name="noise")
    ones = keras.Input((1,), name="ones")

    const = keras.Sequential([
        keras.layers.Dense(const_size * const_size * 128, use_bias=False, name="const"),
        keras.layers.Reshape((const_size, const_size, 128)),
    ], name="const")(ones)

    w = Map(size=128, num_layers=3)(z)
    noise = tf.expand_dims(noise_, axis=-1)

    x = AddNoise()((const, noise))
    x = AdaNorm(exclude_mean=exclude_mean)(x)
    x = Style(64, upsampling=False)((x, w[:, 0], noise))  # 7^2
    for i in range(1, n_style_block):
        x = Style(64)((x, w[:, i], noise))
    o = keras.layers.Conv2D(img_shape[-1], 5, 1, "same", activation=keras.activations.tanh)(x)

    g = keras.Model([ones, z, noise_], o, name="generator")
    g.summary()
    return g, n_style_block


class StyleGAN(keras.Model):
    def __init__(self, img_shape, latent_dim, exclude_mean,
                 summary_writer=None, lr=0.0002, beta1=0.5, beta2=0.99, lambda_=10,):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.exclude_mean = exclude_mean
        self.lambda_ = lambda_

        self.g, self.n_style_block = get_generator(latent_dim, img_shape, exclude_mean)
        self.g.summary()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)

        self.summary_writer = summary_writer
        self._train_step = 0

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs[0], np.ndarray):
            inputs = [tf.convert_to_tensor(i) for i in inputs]
        return self.g.call(inputs, training=training)

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

        model.add(Conv2D(64, 3, 2, "valid"))
        model.add(InstanceNormalization(exclude_mean=exclude_mean))
        model.add(Flatten())
        model.add(keras.layers.Dense(128))
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

    def get_inputs(self, n):
        available_z = [tf.random.normal((n, 1, self.latent_dim)) for _ in range(2)]
        z = tf.concat([available_z[np.random.randint(0, len(available_z))] for _ in range(self.n_style)], axis=1)

        noise = tf.random.normal((n, self.img_shape[0], self.img_shape[1]))
        return [tf.ones((n, 1)), z, noise]

    def train_d(self, img):
        n = len(img)

        with tf.GradientTape() as tape:
            gimg = self.call(self.get_inputs(n), training=False)
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
        with tf.GradientTape() as tape:
            gimg = self.call(self.get_inputs(n), training=True)
            pred_fake = self.d.call(gimg, training=False)
            w_distance = tf.reduce_mean(-pred_fake)  # minimize W distance
        grads = tape.gradient(w_distance, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))

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