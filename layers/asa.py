import tensorflow as tf
from tensorflow.keras import layers


class ASAModule(layers.Layer):
 
    def __init__(self, n_channels, **kwargs):
        super(ASAModule, self).__init__(**kwargs)
        self.n_channels = n_channels
 
        # MLP lapisan pertama: 2C → 2C, aktivasi ReLU
        self.fc1 = layers.Dense(
            units      = n_channels * 2,
            activation = 'relu',
            name       = 'asa_fc1'
        )
 
        # MLP lapisan kedua: 2C → C, aktivasi Sigmoid
        self.fc2 = layers.Dense(
            units      = n_channels,
            activation = 'sigmoid',
            name       = 'asa_fc2'
        )
        
        
    def build(self, input_shape):
        self.fc1.build((None, self.n_channels * 2))
        self.fc2.build((None, self.n_channels * 2))

        super().build(input_shape)
        
 
    def call(self, x, training=False):
        gap = tf.reduce_mean(x, axis=1)
        gmp = tf.reduce_max(x, axis=1)
 
        # Konkatenasi GAP dan GMP
        z = tf.concat([gap, gmp], axis=-1)
 
        # MLP
        z = self.fc1(z)
        w_attn = self.fc2(z)
 
        # Pembobotan pointwise via broadcasting
        w_attn = tf.expand_dims(w_attn, axis=1)

        x_out  = x * w_attn
        return x_out
 
    def get_config(self):
        config = super().get_config()
        config.update({'n_channels': self.n_channels})
        return config