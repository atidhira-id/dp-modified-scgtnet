import tensorflow as tf
from tensorflow.keras import layers, Model


class MSCModule(layers.Layer):
    """
    Multi-Scale Convolution Module.
 
    Langkah:
      1. Reshape input (B, T, C) → (B, T, C, 1)
      2. Tiga cabang Conv2D paralel dengan kernel 3, 5, 7
      3. Konkatenasi ketiga output → (B, T, C, 3F)
      4. Reshape → (B, T, C*3F)
 
    Input  : (B, T, C)
    Output : (B, T, C*3F)
    """
 
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
 
    def call(self, x, training=False):
        # x shape: (B, T, C)
 
        # Langkah 1 — Reshape untuk Conv2D
        x_reshaped = tf.expand_dims(x, axis=-1)  # (B, T, C, 1)
 
        # Langkah 2 — Tiga cabang Conv2D paralel
        y3 = self.conv3(x_reshaped)              # (B, T, C, F)
        y5 = self.conv5(x_reshaped)              # (B, T, C, F)
        y7 = self.conv7(x_reshaped)              # (B, T, C, F)
 
        # Langkah 3 — Konkatenasi sepanjang dimensi filter
        y_concat = tf.concat([y3, y5, y7], axis=-1)  # (B, T, C, 3F)
 
        # Langkah 4 — Reshape kembali untuk Bi-GRU
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