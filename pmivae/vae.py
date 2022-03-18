import tensorflow as tf
from tensorflow.keras import Model, layers
from dataclasses import dataclass, field
from typing import Any, List, Tuple
import numpy as np


@dataclass
class ConfigVAE:
    optimizer: Any = "adam"
    loss: Any = "mean_squared_error"
    metrics: List = field(default_factory=lambda: [])
    callbacks: List = field(default_factory=lambda: [])
    epochs: int = 200
    batch_size: int = 32
    validation_split: float = 0.0
    verbose: int = 1
    filters: List[int] = field(default_factory=lambda: [])
    kernels: List[float] = field(default_factory=lambda: [])
    activation: str = "relu"
    output_activation: str = "sigmoid"
    neurons: List[int] = field(default_factory=lambda: [10])
    dropout_conv: List[float] = None
    dropout_fc: List[float] = None
    latent_dimension: int = 1
    input_shape: Tuple = None

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        epsilon.set_shape(z_mean.shape)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class KLDivergenceLayer(layers.Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * tf.keras.backend.sum(
            1 + log_var - tf.keras.backend.square(mu) -
            tf.keras.backend.exp(log_var), axis=-1)

        self.add_loss(tf.keras.backend.mean(kl_batch), inputs=inputs)
        return inputs

class VariationalAutoEncoder:

    def _create_auto_encoder(self, input_shape):
        x = enc_input = tf.keras.Input(shape=input_shape)

        for i, f in enumerate(self._config.filters):
            k = self._config.kernels[i] if isinstance(self._config.kernels, list) else self._config.kernels
            x = layers.Conv2D(f, kernel_size=k, padding='same', activation=self._config.activation)(x)
            x = layers.MaxPool2D(2, strides=2)(x)
            if self._config.dropout_conv is not None:
                x = layers.Dropout(rate=self._config.dropout_conv[i])(x)

        shape_before_flat = None
        if len(self._config.filters) > 0:
            shape_before_flat = list(filter(None, x.get_shape().as_list()))
            x = layers.Flatten()(x)

        for i, n in enumerate(self._config.neurons):
            x = layers.Dense(n, activation=self._config.activation)(x)
            if self._config.dropout_fc is not None:
                x = layers.Dropout(rate=self._config.dropout_fc[i])(x)

        z_mean_layer = layers.Dense(self._config.latent_dimension, name="z_mean")
        z_log_var_layer = layers.Dense(self._config.latent_dimension, name="z_log_var")
        z_mean = z_mean_layer(x)
        z_log_var = z_log_var_layer(x)
        z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
        x = [z_mean, z_log_var, Sampling()([z_mean, z_log_var])]

        m_encoder = Model(enc_input, x, name='encoder')
        dec_input = tf.keras.Input(shape=(self._config.latent_dimension,))
        x = dec_input

        for i, n in reversed(list(enumerate(self._config.neurons))):
            x = layers.Dense(n, activation=self._config.activation)(x)
            if self._config.dropout_fc is not None:
                x = layers.Dropout(rate=self._config.dropout_fc[i])(x)

        if len(self._config.filters) > 0:
            x = layers.Dense(units=np.prod(shape_before_flat), activation=self._config.activation)(x)
            x = layers.Reshape(target_shape=shape_before_flat)(x)

            for i, f in reversed(list(enumerate(self._config.filters))):
                k = self._config.kernels[i] if isinstance(self._config.kernels, list) else self._config.kernels
                x = layers.Conv2DTranspose(f, kernel_size=k, strides=2, padding='same',
                                           activation=self._config.activation)(x)
                if self._config.dropout_conv is not None:
                    x = layers.Dropout(rate=self._config.dropout_conv[i])(x)

            x = layers.Conv2DTranspose(filters=list(filter(None, enc_input.get_shape().as_list()))[2], kernel_size=1,
                                       strides=1, padding='same', activation=self._config.output_activation)(x)
        else:
            x = layers.Dense(units=list(filter(None, enc_input.get_shape().as_list()))[0],
                             activation=self._config.output_activation)(x)

        m_decoder = Model(dec_input, x, name='decoder')
        enc_output = m_decoder(m_encoder(enc_input)[2])
        m_global = Model(enc_input, enc_output, name='vae')

        return m_global, m_encoder, m_decoder

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        self._model, self._encoder, self._decoder = self._create_auto_encoder(self._config.input_shape)
        self._model.compile(optimizer=self._config.optimizer, loss=self._config.loss, metrics=self._config.metrics)

        fit_args = {
            "epochs": self._config.epochs,
            "batch_size": self._config.batch_size,
            "callbacks": self._config.callbacks,
            "validation_split": self._config.validation_split,
            "verbose": self._config.verbose
        }

        if x_val is not None and y_val is not None:
            fit_args["validation_data"] = (x_val, y_val)

        self._model.fit(x_train, y_train, **fit_args)
        self._fitted = True

    def encode(self, x):
        if not self._fitted:
            raise RuntimeError("The fit method must be called before encode.")
        return self._encoder.predict(x)

    def decode(self, x):
        if not self._fitted:
            raise RuntimeError("The fit method must be called before decode.")
        return self._decoder.predict(x)

    def __init__(self, config: ConfigVAE):
        self._config = config
        self._model = None
        self._encoder = None
        self._decoder = None
        self._fitted = False
