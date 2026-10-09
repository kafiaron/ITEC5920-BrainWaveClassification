import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt


# Zero-phase Butterworth band-pass
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
def bandpass(X, band, fs, order=4):
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=1).astype(np.float32)


def _covariances(X):
    return np.einsum("ntc,ntd->ncd", X, X) / X.shape[1]


def _inv_sqrt(R):
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, 1e-10 * w.max())
    return (V / np.sqrt(w)) @ V.T


# Euclidean alignment: whiten by the inverse square root of the mean covariance
# Unsupervised, fit on training trials only
# Reference: He & Wu (2020)
class EuclideanAligner:

    def fit(self, X):
        self.R_ = _inv_sqrt(_covariances(X).mean(0))
        return self

    def transform(self, X):
        return (X @ self.R_).astype(np.float32)


# Normalize using train set stats to avoid data leakage
# Reference: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=(0, 1), keepdims=True)
        self.std_ = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)


# CSP with shrinkage-regularized, trace-normalized class covariances
# Reference: Müller-Gerking et al. (1999)
class CSP:

    def __init__(self, n_components=6, reg=0.05):
        self.n_components = n_components - n_components % 2
        self.reg = reg

    def _class_cov(self, X):
        C = _covariances(X)
        C = (C / np.trace(C, axis1=1, axis2=2)[:, None, None]).mean(0)
        n = C.shape[0]
        return (1 - self.reg) * C + self.reg * np.trace(C) / n * np.eye(n)

    def fit(self, X, y):
        C0, C1 = self._class_cov(X[y == 0]), self._class_cov(X[y == 1])
        _, V = eigh(C0, C0 + C1)
        h = self.n_components // 2
        self.W_ = np.concatenate([V[:, :h], V[:, -h:]], axis=1)
        return self

    def project(self, X):
        return (X @ self.W_).astype(np.float32)

    def logvar(self, X):
        v = self.project(X).var(axis=1)
        return np.log(v / v.sum(axis=1, keepdims=True))
