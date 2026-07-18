import tensorflow as tf
from tensorflow.keras import layers, Model


class BiGRUModule(layers.Layer):
    """
    Bidirectional GRU Module.
 
    Langkah:
      1. Bidirectional GRU — forward + backward secara paralel
      2. Konkatenasi output forward dan backward → (B, T, G*2)
      3. Layer Normalization
 
    Input  : (B, T, C*3F)
    Output : (B, T, G*2)
    """
 
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
            merge_mode = 'concat',    # output: (B, T, G*2)
            name       = 'bigru'
        )
 
        # Layer Normalization setelah Bi-GRU
        self.layer_norm = layers.LayerNormalization(
            epsilon = 1e-6,
            name    = 'bigru_layernorm'
        )
 
    def call(self, x, training=False):
        # x shape: (B, T, C*3F)
 
        # Langkah 1 & 2 — Bi-GRU + konkatenasi otomatis
        h = self.bigru(x, training=training)     # (B, T, G*2)
 
        # Langkah 3 — Layer Normalization
        h_norm = self.layer_norm(h)              # (B, T, G*2)
 
        return h_norm
 
    def get_config(self):
        config = super().get_config()
        config.update({
            'gru_units'   : self.gru_units,
            'dropout_rate': self.dropout_rate
        })
        return config
 