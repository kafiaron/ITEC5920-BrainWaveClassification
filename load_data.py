import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

import config as cfg


def _labels(subj):
    y = loadmat(cfg.P4_DIR / f"true_labels_{subj}.mat")["true_y"].ravel().astype(np.int64)
    return y - y.min()


# Load raw 118-channel recordings, band-pass the continuous signal, then epoch
# Official test set = trials without public labels in the raw markers
# Reference: https://www.bbci.de/competition/iii/desc_IVa.html
def _load_raw(subj, y_true):
    path = cfg.CACHE_DIR / f"raw_{subj}.npz"
    if path.exists():
        d = np.load(path)
        return d["X"], [str(c) for c in d["clab"]], d["test_idx"]

    m = loadmat(cfg.RAW_DIR / f"data_set_IVa_{subj}.mat")
    fs = int(np.ravel(m["nfo"]["fs"][0, 0])[0])
    if fs != cfg.FS:
        raise ValueError(f"{subj}: expected {cfg.FS} Hz data, got {fs} Hz")

    clab = [str(c[0]) for c in m["nfo"]["clab"][0, 0].ravel()]
    pos = m["mrk"][0, 0]["pos"].ravel().astype(np.int64) - 1
    y_mrk = m["mrk"][0, 0]["y"].ravel()

    # The official competition test set is exactly the trials without public labels.
    test_idx = np.flatnonzero(np.isnan(y_mrk))
    known = ~np.isnan(y_mrk)
    if not np.array_equal(y_mrk[known].astype(np.int64) - 1, y_true[known]):
        raise ValueError(f"{subj}: true_labels file disagrees with raw markers")

    cnt = m["cnt"].astype(np.float64) * 0.1  # to microvolts
    sos = butter(4, cfg.BROAD_BAND, btype="bandpass", fs=fs, output="sos")
    cnt = sosfiltfilt(sos, cnt, axis=0)

    a, b = int(cfg.EPOCH_WINDOW[0] * fs), int(cfg.EPOCH_WINDOW[1] * fs)
    if pos.max() + b > len(cnt):
        raise ValueError(f"{subj}: last epoch exceeds recording length")
    X = np.stack([cnt[p + a:p + b] for p in pos]).astype(np.float32)

    cfg.CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(path, X=X, clab=np.array(clab), test_idx=test_idx)
    return X, clab, test_idx


# Load one subject as dict(X [trials, time, channels], y, test_idx, padded, clab)
# channels="p4" loads the legacy preprocessed features from the course
def load_subject(subj, channels="all"):
    y = _labels(subj)

    if channels == "p4":
        f = loadmat(cfg.P4_DIR / f"features_{subj}_440_0.mat")["features"]
        X = np.transpose(f, (2, 1, 0)).astype(np.float32)
        _, _, test_idx = _load_raw(subj, y)
        return dict(X=X, y=y, test_idx=test_idx, padded=False,
                    clab=[f"p4_{i}" for i in range(X.shape[2])])

    X, clab, test_idx = _load_raw(subj, y)
    if channels != "all":
        names = cfg.CHANNEL_SETS[channels]
        X = X[:, :, [clab.index(c) for c in names]]
        clab = list(names)
    return dict(X=X, y=y, test_idx=test_idx, padded=True, clab=clab)


def crop(X, padded):
    if not padded:
        return X
    a = int((cfg.CROP_WINDOW[0] - cfg.EPOCH_WINDOW[0]) * cfg.FS)
    b = int((cfg.CROP_WINDOW[1] - cfg.EPOCH_WINDOW[0]) * cfg.FS)
    return X[:, a:b]
