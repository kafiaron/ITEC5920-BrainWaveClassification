import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as L
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

TEMPORAL_KERNEL = 50  # fs / 2 at 100 Hz, following EEGNet
SEPARABLE_KERNEL = 16
TCFORMER_KERNELS = (8, 13, 26)  # 20/32/64 samples at 250 Hz, rescaled to 100 Hz
L2 = 1e-3

# Every deviation from the cited architectures, reported in manifest.json (reviewer R2-9)
ARCHITECTURE_NOTES = {
    "eegnet": "EEGNet-8,2 (Lawhern et al. 2018) at 100 Hz: temporal kernel 50 (fs/2), separable "
              "kernel 16, pools 4 and 8, max-norm 1.0 on depthwise and 0.25 on the dense output. "
              "Sigmoid output for the binary task instead of softmax.",
    "ctnet": "CTNet (Zhao et al. 2024): EEGNet front-end, learnable positional embedding, "
             "Transformer encoder, global residual (tokens + encoder(tokens)), flatten, dense. "
             "Deviations: 1 encoder layer instead of 6 and 2-4 heads, because of 280 trials per "
             "subject; second pool 4 so 200 samples give 12 tokens; sigmoid output.",
    "tcformer": "TCFormer (Altaheri et al. 2025): multi-kernel temporal convolutions, depthwise "
                "spatial conv, positional embedding, Transformer with GELU FFN, 1x1 reduction "
                "(BN + SiLU), concatenation with the conv tokens, TCN head (2 residual causal "
                "blocks, kernel 4, dilations 1 and 2), last time step, dense with max-norm 0.25. "
                "Deviations: standard multi-head attention instead of grouped-query attention, "
                "1 encoder layer, kernels rescaled from 250 Hz to 100 Hz.",
    "hybrid": "Proposed. Temporal branch: the CTNet configuration above on Euclidean-aligned EEG, "
              "ending in a 1-unit dense logit. Geometric branch: tangent-space features at the identity "
              "of 8-30 Hz OAS covariances re-centred by the subject's Riemannian mean (training trials "
              "only), z-scored, dropout 0.2, 1-unit dense logit with L2 1e-3. Output: sigmoid of the "
              "summed logits. Pretraining on other subjects uses the same per-subject alignment.",
    "siamese": "Siamese network with a shared CTNet encoder and a 32-d ELU embedding. "
               "Similarity = sigmoid(w * |e_a - e_b| + b), trained with binary cross-entropy. "
               "Pairs: k same-class and k different-class partners per anchor, resampled every "
               "epoch; augmented copies of the anchor's own trial are never its positives. "
               "Then a dropout + dense sigmoid head on the frozen encoder, followed by optional "
               "fine-tuning of the encoder (BatchNorm frozen) kept only if validation loss improves.",
}


class PositionalEmbedding(L.Layer):
    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(input_shape[1], input_shape[2]),
                                   initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        return x + self.pos


class AbsDiff(L.Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])


# EEGNet-style front-end: temporal conv, depthwise spatial conv, separable conv
# Reference: Lawhern et al. (2018)
def _conv_frontend(inp, T, C, F1, D, F2, dropout, pool2, kernels=(TEMPORAL_KERNEL,)):
    x = L.Reshape((T, C, 1))(inp)
    branches = [L.Conv2D(F1, (k, 1), padding="same", use_bias=False)(x) for k in kernels]
    x = branches[0] if len(branches) == 1 else L.Concatenate()(branches)
    x = L.BatchNormalization()(x)
    x = L.DepthwiseConv2D((1, C), depth_multiplier=D, use_bias=False,
                          depthwise_constraint=max_norm(1.0))(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("elu")(x)
    x = L.AveragePooling2D((4, 1))(x)
    x = L.Dropout(dropout)(x)
    x = L.SeparableConv2D(F2, (SEPARABLE_KERNEL, 1), padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("elu")(x)
    x = L.AveragePooling2D((pool2, 1))(x)
    x = L.Dropout(dropout)(x)
    return L.Reshape((-1, F2))(x)


# Pre-norm Transformer encoder layer
# Reference: Zhao et al. (2024)
def _encoder_layer(x, d, heads, dropout):
    h = L.LayerNormalization(epsilon=1e-6)(x)
    h = L.MultiHeadAttention(heads, key_dim=d // heads, dropout=dropout)(h, h)
    x = L.Add()([x, L.Dropout(dropout)(h)])
    h = L.LayerNormalization(epsilon=1e-6)(x)
    h = L.Dense(2 * d, activation="gelu", kernel_regularizer=l2(L2))(h)
    h = L.Dropout(dropout)(h)
    h = L.Dense(d, kernel_regularizer=l2(L2))(h)
    return L.Add()([x, L.Dropout(dropout)(h)])


def _ctnet_features(inp, T, C, F1, D, F2, heads, dropout):
    tokens = _conv_frontend(inp, T, C, F1, D, F2, dropout, pool2=4)
    encoded = _encoder_layer(PositionalEmbedding()(tokens), F2, heads, dropout)
    return L.Flatten()(L.Add()([tokens, encoded]))


# Residual causal TCN block
# Reference: Bai et al. (2018); Altaheri et al. (2025)
def _tcn_block(x, filters, dilation, dropout):
    h = x
    for _ in range(2):
        h = L.Conv1D(filters, 4, padding="causal", dilation_rate=dilation,
                     kernel_regularizer=l2(L2))(h)
        h = L.BatchNormalization()(h)
        h = L.Activation("elu")(h)
        h = L.Dropout(dropout)(h)
    if x.shape[-1] != filters:
        x = L.Conv1D(filters, 1)(x)
    return L.Activation("elu")(L.Add()([x, h]))


# Baseline CNN
# Reference: Lawhern et al. (2018)
def build_eegnet(T, C, F1=8, D=2, F2=16, dropout=0.5, **_):
    inp = L.Input((T, C))
    x = L.Flatten()(_conv_frontend(inp, T, C, F1, D, F2, dropout, pool2=8))
    out = L.Dense(1, activation="sigmoid", kernel_constraint=max_norm(0.25))(x)
    return Model(inp, out, name="eegnet")


# CTNet-inspired hybrid
# Reference: Zhao et al. (2024)
def build_ctnet(T, C, F1=8, D=2, F2=16, heads=2, dropout=0.5, **_):
    inp = L.Input((T, C))
    x = L.Dropout(dropout)(_ctnet_features(inp, T, C, F1, D, F2, heads, dropout))
    return Model(inp, L.Dense(1, activation="sigmoid")(x), name="ctnet")


# TCFormer-inspired model
# Reference: Altaheri et al. (2025)
def build_tcformer(T, C, F1=8, D=2, F2=16, heads=2, dropout=0.5, **_):
    inp = L.Input((T, C))
    tokens = _conv_frontend(inp, T, C, F1, D, F2, dropout, pool2=4, kernels=TCFORMER_KERNELS)
    encoded = _encoder_layer(PositionalEmbedding()(tokens), F2, heads, dropout)
    reduced = L.Activation("silu")(L.BatchNormalization()(L.Conv1D(F2 // 2, 1)(encoded)))
    x = L.Concatenate()([tokens, reduced])
    for dilation in (1, 2):
        x = _tcn_block(x, F2, dilation, dropout)
    x = L.Lambda(lambda t: t[:, -1, :])(x)
    out = L.Dense(1, activation="sigmoid", kernel_constraint=max_norm(0.25))(x)
    return Model(inp, out, name="tcformer")


# Siamese-CTNet: returns (pair model, shared encoder, classifier sharing the encoder)
# Reference: https://keras.io/examples/vision/siamese_network/
def build_siamese(T, C, F1=8, D=2, F2=16, heads=2, dropout=0.5, emb_dim=32, **_):
    inp = L.Input((T, C))
    features = _ctnet_features(inp, T, C, F1, D, F2, heads, dropout)
    encoder = Model(inp, L.Dense(emb_dim, activation="elu")(features), name="encoder")

    a, b = L.Input((T, C)), L.Input((T, C))
    similarity = L.Dense(1, activation="sigmoid")(AbsDiff()([encoder(a), encoder(b)]))
    pair = Model([a, b], similarity, name="siamese_pair")

    single = L.Input((T, C))
    head = L.Dense(1, activation="sigmoid")(L.Dropout(dropout)(encoder(single)))
    return pair, encoder, Model(single, head, name="siamese_classifier")


# Dual-aligned hybrid: a CTNet temporal branch on Euclidean-aligned EEG and a linear geometric
# branch on tangent-space features of Riemannian re-centred covariances. Both inputs are aligned
# per subject, so the whole model can be pretrained on other subjects and fine-tuned on few trials.
def build_hybrid(T, C, n_tangent, F1=8, D=2, F2=16, heads=2, dropout=0.5, tangent_l2=1e-3, **_):
    raw, tangent = L.Input((T, C)), L.Input((n_tangent,))
    temporal = L.Dense(1)(L.Dropout(dropout)(_ctnet_features(raw, T, C, F1, D, F2, heads, dropout)))
    geometric = L.Dense(1, kernel_regularizer=l2(tangent_l2))(L.Dropout(0.2)(tangent))
    out = L.Activation("sigmoid")(L.Add()([temporal, geometric]))
    return Model([raw, tangent], out, name="hybrid")


BUILDERS = {"eegnet": build_eegnet, "ctnet": build_ctnet, "tcformer": build_tcformer,
            "hybrid": build_hybrid}
