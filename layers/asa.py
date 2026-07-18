import tensorflow as tf
from tensorflow.keras import layers, Model


class ASAModule(layers.Layer):
    """
    Adaptive Spatial Attention Module.
 
    Langkah:
      1. GAP  — Global Average Pooling sepanjang dimensi T
      2. GMP  — Global Maximum Pooling sepanjang dimensi T
      3. Concat GAP dan GMP → (B, 2C)
      4. MLP  — FC(2C→2C, ReLU) → FC(2C→C, Sigmoid)
      5. Pembobotan pointwise pada input X
 
    Input  : (B, T, C)
    Output : (B, T, C)
    """
 
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
 
    def call(self, x, training=False):
        # x shape: (B, T, C)
 
        # GAP: rata-rata sepanjang dimensi T
        # (B, C)
        gap = tf.reduce_mean(x, axis=1)          
 
        # GMP: maksimum sepanjang dimensi T
        # (B, C)
        gmp = tf.reduce_max(x, axis=1)           
 
        # Konkatenasi GAP dan GMP
        # (B, 2C)
        z = tf.concat([gap, gmp], axis=-1)       
 
        # MLP
        # (B, 2C)
        z = self.fc1(z)
        # (B, C)
        w_attn = self.fc2(z)                     
 
        # Pembobotan pointwise via broadcasting
        # w_attn: (B, C) → expand → (B, 1, C) → broadcast ke (B, T, C)
        # (B, 1, C)
        w_attn = tf.expand_dims(w_attn, axis=1)  

        # (B, T, C)
        x_out  = x * w_attn                      
        return x_out
 
    def get_config(self):
        config = super().get_config()
        config.update({'n_channels': self.n_channels})
        return config