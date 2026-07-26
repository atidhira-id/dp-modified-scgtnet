import tensorflow as tf
from tensorflow.keras import layers


class BiGRUModule(layers.Layer):
 
    def __init__(self, gru_units=64, dropout_rate=0.1, **kwargs):
        super(BiGRUModule, self).__init__(**kwargs)
        self.gru_units    = gru_units
        self.dropout_rate = dropout_rate
 
        # Bidirectional GRU
        # return_sequences=True agar seluruh langkah waktu dikembalikan
        # merge_mode='concat' untuk konkatenasi forward dan backward
        self.bigru = layers.Bidirectional(
            layers.GRU(
                units            = gru_units,
                return_sequences = True,
                dropout          = dropout_rate,
                recurrent_dropout= 0.0,
                name             = 'gru_layer'
            ),
            merge_mode = 'concat',
            name       = 'bigru'
        )
 
        # Layer Normalization setelah Bi-GRU
        self.layer_norm = layers.LayerNormalization(
            epsilon = 1e-6,
            name    = 'bigru_layernorm'
        )
 
 
    def build(self, input_shape):
        self.bigru.build(input_shape)
        output_shape = (
            input_shape[0],
            input_shape[1],
            self.gru_units * 2
        )
        self.layer_norm.build(output_shape)
        super().build(input_shape)
        
 
    def call(self, x, training=None):
        # Bi-GRU + konkatenasi otomatis
        h = self.bigru(x, training=training)
 
        # Layer Normalization
        h_norm = self.layer_norm(h)
 
        return h_norm
 
    def get_config(self):
        config = super().get_config()
        config.update({
            'gru_units'   : self.gru_units,
            'dropout_rate': self.dropout_rate
        })
        return config
 