import os
from dataclasses import asdict, dataclass

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping

from models import BUILDERS, build_siamese


@dataclass
class TrainConfig:
    lr: float = 1e-3
    lr_finetune: float = 1e-4
    encoder_finetune_factor: float = 0.1
    epochs: int = 150
    patience: int = 20
    epochs_no_val: int = 60
    batch: int = 32
    pairs_per_anchor: int = 4

    @classmethod
    def quick(cls):
        return cls(epochs=15, patience=4, epochs_no_val=15)

    @property
    def tag(self):
        return f"e{self.epochs}p{self.patience}"

    def as_dict(self):
        return asdict(self)


def set_seed(seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)


# XLA is disabled: Keras enables it on GPU by default, which recompiles every new model and
# leaks host memory over the thousands of models built in a full run
def _compile(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy"], jit_compile=False)


# Early stopping on the inner validation split only, never the test fold.
# Keras 3 restores the best weights at the end of training even if stopping never triggers.
# Reference: https://keras.io/api/callbacks/early_stopping/
def _fit(model, X, y, val, lr, tc):
    _compile(model, lr)
    if val is None:
        model.fit(X, y, epochs=tc.epochs_no_val, batch_size=tc.batch, shuffle=True, verbose=0)
        return dict(best_epoch=tc.epochs_no_val, epochs_run=tc.epochs_no_val)
    es = EarlyStopping(monitor="val_loss", patience=tc.patience, restore_best_weights=True)
    hist = model.fit(X, y, validation_data=val, epochs=tc.epochs, batch_size=tc.batch,
                     shuffle=True, callbacks=[es], verbose=0)
    return dict(best_epoch=es.best_epoch + 1, epochs_run=len(hist.history["loss"]))


# k same-class and k different-class partners per anchor. Positives never come from
# the anchor's own source trial (its augmented copies).
def pair_indices(y, src, k, rng):
    a, b, labels = [], [], []
    for i in range(len(y)):
        same = np.flatnonzero((y == y[i]) & (src != src[i]))
        diff = np.flatnonzero(y != y[i])
        for pool, label in ((same, 1.0), (diff, 0.0)):
            if len(pool) == 0:
                continue
            a.extend([i] * k)
            b.extend(rng.choice(pool, size=k, replace=len(pool) < k))
            labels.extend([label] * k)
    return np.array(a), np.array(b), np.array(labels, dtype=np.float32)


def _pair_dataset(Xt, idx, batch, shuffle, seed=0):
    a, b, labels = idx
    ds = tf.data.Dataset.from_tensor_slices((a, b, labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed)
    return ds.batch(batch).map(lambda i, j, t: ((tf.gather(Xt, i), tf.gather(Xt, j)), t))


def _set_encoder_trainable(encoder, trainable):
    encoder.trainable = trainable
    if trainable:
        for layer in encoder.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


class Net:
    def __init__(self, name, T, C, hp):
        self.name = name
        if name == "siamese":
            self.pair, self.encoder, self.classifier = build_siamese(T, C, **hp)
        else:
            self.classifier = BUILDERS[name](T, C, **hp)

    def fit(self, X, y, src, val, tc, lr, rng):
        if self.name != "siamese":
            return _fit(self.classifier, X, y, val, lr, tc)
        info = self._fit_pairs(X, y, src, val, tc, lr, rng)
        _set_encoder_trainable(self.encoder, False)
        _fit(self.classifier, X, y, val, lr, tc)
        if val is not None:
            self._finetune(X, y, val, tc, lr)
        return info

    # Stage 1: metric learning, pairs resampled every epoch, manual early stopping on
    # fixed pairs built from the inner validation split
    def _fit_pairs(self, X, y, src, val, tc, lr, rng):
        self.encoder.trainable = True
        self.pair.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
                          jit_compile=False)
        Xt = tf.constant(X)
        val_ds = None
        if val is not None:
            Xv, yv = val
            val_ds = _pair_dataset(tf.constant(Xv), pair_indices(yv, np.arange(len(yv)),
                                                                  tc.pairs_per_anchor, rng),
                                   tc.batch, shuffle=False)
        n_epochs = tc.epochs if val is not None else tc.epochs_no_val
        best_loss, best_weights, best_epoch, wait, epoch = np.inf, None, n_epochs, 0, 0
        for epoch in range(1, n_epochs + 1):
            idx = pair_indices(y, src, tc.pairs_per_anchor, rng)
            self.pair.fit(_pair_dataset(Xt, idx, tc.batch, True, int(rng.integers(2**31))),
                          epochs=1, verbose=0)
            if val_ds is None:
                continue
            loss = self.pair.evaluate(val_ds, verbose=0)
            if loss < best_loss:
                best_loss, best_weights, best_epoch, wait = loss, self.pair.get_weights(), epoch, 0
            else:
                wait += 1
                if wait >= tc.patience:
                    break
        if best_weights is not None:
            self.pair.set_weights(best_weights)
        return dict(best_epoch=best_epoch, epochs_run=epoch)

    # Stage 2b: unfreeze the encoder (BatchNorm stays frozen) at a lower learning rate,
    # keep the result only if validation loss improves
    def _finetune(self, X, y, val, tc, lr):
        before = self.classifier.evaluate(*val, verbose=0)[0]
        weights = self.classifier.get_weights()
        _set_encoder_trainable(self.encoder, True)
        _fit(self.classifier, X, y, val, lr * tc.encoder_finetune_factor, tc)
        if self.classifier.evaluate(*val, verbose=0)[0] >= before:
            self.classifier.set_weights(weights)

    def predict(self, X):
        return self.classifier.predict(X, batch_size=256, verbose=0).ravel()

    def save(self, path):
        self.classifier.save_weights(str(path))

    def load(self, path):
        self.classifier.load_weights(str(path))

    def count_params(self):
        return self.classifier.count_params()


# Model selection rule: highest validation accuracy, then lowest validation log-loss,
# then grid order (ties keep the earlier configuration)
def validation_score(net, X, y):
    p = np.clip(net.predict(X), 1e-6, 1 - 1e-6)
    return accuracy_score(y, p >= 0.5), -log_loss(y, p, labels=[0, 1])


# Leave-one-subject-out pretraining: source subjects only, 10% source validation split.
# Weights are cached so every fold, repeat and augmentation condition reuses them.
def pretrain(name, hp, Xs, ys, tc, seed, path):
    if path.exists():
        return
    set_seed(seed)
    rng = np.random.default_rng(seed)
    X0 = Xs[0] if isinstance(Xs, list) else Xs
    take = (lambda i: [a[i] for a in Xs]) if isinstance(Xs, list) else (lambda i: Xs[i])
    fit, val = next(StratifiedShuffleSplit(1, test_size=0.1, random_state=seed).split(X0, ys))
    net = Net(name, X0.shape[1], X0.shape[2], hp)
    net.fit(take(fit), ys[fit], np.arange(len(fit)), (take(val), ys[val]), tc, tc.lr, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.weights.h5")
    net.save(tmp)
    os.replace(tmp, path)  # atomic, so parallel jobs never read a half-written file
