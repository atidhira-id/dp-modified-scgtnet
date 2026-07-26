import tensorflow as tf
from tensorflow.keras import layers, Model
from layers.asa import ASAModule
from layers.msc import MSCModule
from layers.bi_gru import BiGRUModule
from layers.transformer import TransformerEncoderModule


def build_modified_scgtnet(
    T             = 40,
    C             = 16,
    n_classes     = 52,
    msc_filters   = 16,
    gru_units     = 64,
    num_heads     = 4,
    dropout_rate  = 0.1,
):
    # d_model = G*2 karena Bi-GRU mengkonkatenasi forward + backward
    d_model = gru_units * 2
 
    # ── Input ─────────────────────────────────────────────
    inputs = tf.keras.Input(
        shape = (T, C),
        name  = 'input_semg'
    )                                            # (B, T, C)
 
    # ── 1. ASA ────────────────────────────────────────────
    x = ASAModule(
        n_channels = C,
        name       = 'asa_module'
    )(inputs)                                    # (B, T, C)
 
    # ── 2. MSC ────────────────────────────────────────────
    x = MSCModule(
        filters = msc_filters,
        name    = 'msc_module'
    )(x)                                       # (B, T, C*3F)
    x = tf.keras.layers.BatchNormalization()(x)
 
    # ── 3. Bi-GRU ─────────────────────────────────────────
    x = BiGRUModule(
        gru_units    = gru_units,
        dropout_rate = dropout_rate,
        name         = 'bigru_module'
    )(x)                                         # (B, T, G*2)
 
    # ── 4. Transformer Encoder ────────────────────────────
    x = TransformerEncoderModule(
        d_model      = d_model,
        num_heads    = num_heads,
        dropout_rate = dropout_rate,
        name         = 'transformer_encoder'
    )(x)                                         # (B, T, G*2)
 
    # ── 5. Global Average Pooling ─────────────────────────
    x = layers.GlobalAveragePooling1D(
        name = 'global_avg_pooling'
    )(x)                                         # (B, G*2)
    
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
 
    # ── 6. Dense + Softmax (Klasifikasi) ─────────────────
    outputs = layers.Dense(
        units      = n_classes,
        activation = 'softmax',
        name       = 'output_classification'
    )(x)                                         # (B, 52)
 
    # ── Build Model ───────────────────────────────────────
    model = Model(
        inputs  = inputs,
        outputs = outputs,
        name    = 'Modified_SCGTNet'
    )
 
    return model
 