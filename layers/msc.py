import tensorflow as tf
from tensorflow.keras import layers, Model


class MSCModule(layers.Layer):
 
    def __init__(self, filters=16, **kwargs):
        super(MSCModule, self).__init__(**kwargs)
        self.filters = filters
 
        # Cabang 1 — kernel 3×1
        self.conv3 = layers.Conv2D(
            filters     = filters,
            kernel_size = (3, 1),
            padding     = 'same',
            activation  = 'relu',
            name        = 'msc_conv3'
        )
 
        # Cabang 2 — kernel 5×1
        self.conv5 = layers.Conv2D(
            filters     = filters,
            kernel_size = (5, 1),
            padding     = 'same',
            activation  = 'relu',
            name        = 'msc_conv5'
        )
 
        # Cabang 3 — kernel 7×1
        self.conv7 = layers.Conv2D(
            filters     = filters,
            kernel_size = (7, 1),
            padding     = 'same',
            activation  = 'relu',
            name        = 'msc_conv7'
        )
        
    
    def build(self, input_shape):
        self.conv3.build((None, input_shape[1], input_shape[2], 1))
        self.conv5.build((None, input_shape[1], input_shape[2], 1))
        self.conv7.build((None, input_shape[1], input_shape[2], 1))

        super().build(input_shape)
 
 
    def call(self, x, training=False):
        # Reshape untuk Conv2D
        x_reshaped = tf.expand_dims(x, axis=-1)  # (B, T, C, 1)
 
        # Tiga cabang Conv2D paralel
        y3 = self.conv3(x_reshaped)              # (B, T, C, F)
        y5 = self.conv5(x_reshaped)              # (B, T, C, F)
        y7 = self.conv7(x_reshaped)              # (B, T, C, F)
 
        # Konkatenasi sepanjang dimensi filter
        y_concat = tf.concat([y3, y5, y7], axis=-1)  # (B, T, C, 3F)
 
        # Reshape kembali untuk Bi-GRU
        B, T, C, F3 = (
            tf.shape(y_concat)[0],
            tf.shape(y_concat)[1],
            tf.shape(y_concat)[2],
            tf.shape(y_concat)[3]
        )
        y_out = tf.reshape(y_concat, (B, T, C * F3))  # (B, T, C*3F)
 
        return y_out
 
 
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config