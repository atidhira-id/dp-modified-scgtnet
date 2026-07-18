import tensorflow as tf
from tensorflow.keras import layers, Model


class TransformerEncoderModule(layers.Layer):
    """
    Transformer Encoder Module.
 
    Komponen:
      A. Multi-Head Attention (MHA)
         - 4 kepala perhatian
         - Residual connection + Layer Normalization
      B. Feed-Forward Network (FFN)
         - FC(d → 4d, Swish) → Dropout → FC(4d → d)
         - Residual connection + Layer Normalization
 
    Input  : (B, T, G*2)
    Output : (B, T, G*2)
    """
 
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
 
    def call(self, x, training=False):
        # x shape: (B, T, G*2) = (B, T, d_model)
 
        # ── A. Multi-Head Attention ───────────────────────
        # Query, Key, Value semuanya dari x (self-attention)
        attn_out = self.mha(
            query   = x,
            key     = x,
            value   = x,
            training= training
        )                                        # (B, T, d_model)
        attn_out = self.dropout_mha(attn_out, training=training)
 
        # Residual connection + Layer Normalization
        x = self.layernorm1(x + attn_out)       # (B, T, d_model)
 
        # ── B. Feed-Forward Network ───────────────────────
        ffn_out = self.ffn_fc1(x)               # (B, T, d_model*4)
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        ffn_out = self.ffn_fc2(ffn_out)         # (B, T, d_model)
 
        # Residual connection + Layer Normalization
        x = self.layernorm2(x + ffn_out)        # (B, T, d_model)
 
        return x
 
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model'     : self.d_model,
            'num_heads'   : self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config
 