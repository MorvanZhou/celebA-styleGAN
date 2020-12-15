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
    def __init__(self, trainable=None, input_shape=None):
        super().__init__()
        self.trainable = trainable
        self.ys, self.yb = None, None
        if input_shape is not None:
            input_shape = [[None, *s] for s in input_shape]
            self.build(input_shape)

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


class Noise(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        self.b1, self.b2 = None, None
        self.scale = None

        if input_shape is not None:
            input_shape = [[None, *s] for s in input_shape]
            self.build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x, noise = inputs
        if noise is None:
            return x
        noise_ = noise[:, :self.b1, :self.b2, :]
        return self.scale * noise_ + x

    def build(self, input_shape):
        x_shape, noise_shape = input_shape
        self.b1, self.b2 = x_shape[1], x_shape[2]
        self.scale = self.add_weight(shape=[1, 1, x_shape[-1]], initializer=keras.initializers.RandomNormal(0, 0.05))


class StyleBlock(Layer):
    def __init__(self, filters, upsampling=True, exclude_mean=False, input_shape=None):
        super().__init__()
        self.filters = filters
        self.upsampling = upsampling
        self.exclude_mean = exclude_mean

        self.ada_mod, self.c, self.up, self.noise = None, None, None, None
        if input_shape is not None:
            x_shape, w_shape, noise_shape = input_shape
            input_shape = [[None, *x_shape], [None, *w_shape], [None, *noise_shape]]
            self.build(input_shape)

    def call(self, inputs, **kwargs):
        x, w, noise = inputs
        x = self.ada_mod((x, w))
        if self.upsampling:
            x = self.up(x)
        x = self.c(x)
        x = self.noise((x, noise))
        x = ReLU()(x)
        x = AdaNorm(exclude_mean=self.exclude_mean)(x)
        return x

    def build(self, input_shape):
        x_shape, w_shape, noise_shape = input_shape
        self.ada_mod = AdaMod(input_shape=(x_shape[1:], w_shape[1:]))
        if self.upsampling:
            self.up = UpSampling2D((2, 2), interpolation="bilinear", input_shape=self.ada_mod.output_shape)
        self.c = Conv2D(self.filters, 3, 1, "same", input_shape=self.up.output_shape if self.upsampling else self.ada_mod.output_shape)
        self.noise = Noise(input_shape=(self.c.output_shape, noise_shape[1:]))


class Generator(keras.Model):
    def __init__(self, latent_dim, img_shape, exclude_mean, const_seed=1):
        super().__init__()
        self.exclude_mean = exclude_mean
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.const_seed = const_seed

        self.const = tf.random.normal([1, 8, 8, 128], 0, 0.05, seed=const_seed)

        f = keras.Sequential([
            Dense(128),
            LeakyReLU(0.2),
            Dense(128),
            LeakyReLU(0.2),
            Dense(128),
            LeakyReLU(0.2),
        ], name="f")

        z1 = keras.Input((latent_dim,))
        z2 = keras.Input((latent_dim,))
        const = keras.Input([8, 8, 128])
        inputs = [const, z1, z2]
        noise_ = keras.Input((img_shape[0], img_shape[1]))
        noise = tf.expand_dims(noise_, axis=-1)
        inputs.append(noise_)

        w1 = f(z1)
        w2 = f(z2)

        x = self.add_noise(const, noise)
        x = AdaNorm(exclude_mean=exclude_mean)(x)
        x = self.style_block(256, x, w1, noise, upsampling=False)  # 8^2
        x = self.style_block(128, x, w1, noise)  # 16^2
        x = self.style_block(128, x, w1, noise)  # 32^2
        x = self.style_block(256, x, w2, noise)  # 64^2
        x = self.style_block(128, x, w2, noise)  # 128^2
        o = Conv2D(3, 7, 1, "same", activation="tanh")(x)
        self.g = keras.Model(inputs, o, name="generator")
        self.built = True

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs[0], np.ndarray):
            inputs = [tf.convert_to_tensor(i) for i in inputs]
        z1, z2, noise = inputs
        o = self.g.call((self.const, z1, z2, noise), training=training)
        return o

    def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
        return self.call(x, training=False)

    @staticmethod
    def add_noise(x, b_noise):
        x_shape = x.shape[1:] if x.shape[0] is None else x.shape
        b_noise_ = b_noise[:, :x_shape[0], :x_shape[1], :]
        scale = tf.Variable(tf.random.normal([1, 1, x.shape[-1]], 0, 0.05))
        return scale * b_noise_ + x

    def style_block(self, filters, x, w, b_noise=None, upsampling=True):
        x = AdaMod()((x, w))
        if upsampling:
            x = UpSampling2D((2, 2), interpolation="bilinear")(x)

        x = Conv2D(filters, 3, 1, "same")(x)
        x = self.add_noise(x, b_noise)
        x = ReLU()(x)
        x = AdaNorm(exclude_mean=self.exclude_mean)(x)
        return x


class Discriminator(keras.Model):
    def __init__(self, exclude_mean, input_shape=None):
        super().__init__()
        self.exclude_mean = exclude_mean
        self.m = None
        if input_shape is not None:
            input_shape = [None, *input_shape]
            self.build(input_shape)

    def call(self, inputs, training=None, mask=None):
        return self.m(inputs, training=training)

    def build(self, input_shape):
        def block(filters, do_norm=True):
            layers = [Conv2D(filters, 3, strides=2, padding='same',
                             kernel_initializer=keras.initializers.RandomNormal(0, 0.02))]
            if do_norm:
                layers.append(InstanceNormalization(exclude_mean=self.exclude_mean))
            layers.append(LeakyReLU(alpha=0.2))
            return layers
        self.m = keras.Sequential([
            Input(shape=input_shape[1:]),   # 128^2
            *block(64, do_norm=False),   # -> 64^2
            *block(128),                 # -> 32^2
            *block(256),                 # -> 16^2
            *block(512),                 # -> 8^2
            Conv2D(64, 3, 2, "valid"),
            InstanceNormalization(exclude_mean=self.exclude_mean),
            Flatten(),
            Dense(128),
            Dense(1),
        ])
        self.built = True


class StyleGAN(keras.Model):
    def __init__(self, img_shape, latent_dim, exclude_mean,
                 summary_writer=None, lr=0.0002, beta1=0.5, beta2=0.99, lambda_=10, ls_loss=False):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.ls_loss = ls_loss
        self.exclude_mean = exclude_mean
        self.lambda_ = lambda_

        self.g = Generator(latent_dim, img_shape, exclude_mean, const_seed=1)
        self.g.summary()
        self.d = Discriminator(exclude_mean, input_shape=img_shape)
        self.d.summary()

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)
        self.d_loss_fun = keras.losses.MeanSquaredError() if self.ls_loss \
            else keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

        self.summary_writer = summary_writer
        self._train_step = 0

    def call(self, inputs, training=None, mask=None):
        return self.g.call(inputs, training=training)

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
        z1 = tf.random.normal((n, self.latent_dim))
        z2 = tf.random.normal((n, self.latent_dim)) if np.random.random() < 0.5 else z1
        noise = tf.random.normal((n, self.img_shape[0], self.img_shape[1]))
        inputs = [z1, z2, noise]
        return inputs

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