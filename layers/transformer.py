import tensorflow as tf
from tensorflow.keras import layers


class TransformerEncoderModule(layers.Layer):
    def __init__(self, d_model, num_heads=4, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderModule, self).__init__(**kwargs)
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.dropout_rate = dropout_rate
 
        # ── Multi-Head Attention ──────────────────────────
        self.mha = layers.MultiHeadAttention(
            num_heads  = num_heads,
            key_dim    = d_model // num_heads,  # dk per head
            value_dim  = d_model // num_heads,  # dv per head
            dropout    = dropout_rate,
            name       = 'multi_head_attention'
        )
        self.dropout_mha  = layers.Dropout(dropout_rate)
        self.layernorm1   = layers.LayerNormalization(epsilon=1e-6, name='ln_mha')
 
        # ── Feed-Forward Network ──────────────────────────
        # Lapisan 1: ekspansi 4x dengan aktivasi Swish
        self.ffn_fc1 = layers.Dense(
            units      = d_model * 4,
            activation = 'swish',
            name       = 'ffn_fc1'
        )
        self.dropout_ffn  = layers.Dropout(dropout_rate)
 
        # Lapisan 2: kompresi kembali ke d_model
        self.ffn_fc2 = layers.Dense(
            units      = d_model,
            activation = None,
            name       = 'ffn_fc2'
        )
        self.layernorm2   = layers.LayerNormalization(epsilon=1e-6, name='ln_ffn')
        
        
    def build(self, input_shape):
        self.mha.build(
            query_shape=input_shape,
            value_shape=input_shape,
            key_shape=input_shape
        )

        self.dropout_mha.build(input_shape)
        self.layernorm1.build(input_shape)
        self.ffn_fc1.build(input_shape)
        self.dropout_ffn.build(
            (input_shape[0], input_shape[1], self.d_model * 4)
        )
        self.ffn_fc2.build(
            (input_shape[0], input_shape[1], self.d_model * 4)
        )
        self.layernorm2.build(input_shape)

        super().build(input_shape)
 
 
    def call(self, x, training=None):
        # ── A. Multi-Head Attention ───────────────────────
        # Query, Key, Value semuanya dari x (self-attention)
        attn_out = self.mha(
            query   = x,
            key     = x,
            value   = x,
            training= training
        )
        
        attn_out = self.dropout_mha(attn_out, training=training)
 
        # Residual connection + Layer Normalization
        x = self.layernorm1(x + attn_out)
 
        # ── B. Feed-Forward Network ───────────────────────
        ffn_out = self.ffn_fc1(x)
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        ffn_out = self.ffn_fc2(ffn_out)
 
        # Residual connection + Layer Normalization
        x = self.layernorm2(x + ffn_out)
 
        return x
 
 
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model'     : self.d_model,
            'num_heads'   : self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config
 