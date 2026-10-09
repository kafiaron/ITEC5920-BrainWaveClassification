from functools import partial

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline

import config as cfg
from load_data import crop
from preprocessing import CSP, bandpass


def _band_view(X, band, padded):
    return crop(bandpass(X, band, cfg.FS), padded) if padded else X


def _lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


# CSP log-variance features + shrinkage LDA
# References:
# Müller-Gerking et al. (1999)
# Ledoit & Wolf (2004) for the shrinkage estimator
class CSPLDA:
    needs_source = False
    uses_covariances = False

    def __init__(self, seed=0, n_components=6):
        self.n_components = n_components

    csp_fit_data = None  # audit only: (X_all, y_all) reproduces CSP fit on every trial

    def fit(self, X, y, padded, source=None):
        Z = _band_view(X, cfg.CSP_BAND, padded)
        Xc, yc = self.csp_fit_data or (X, y)
        self.csp = CSP(self.n_components).fit(_band_view(Xc, cfg.CSP_BAND, padded), yc)
        self.lda = _lda().fit(self.csp.logvar(Z), y)
        return self

    def predict_proba(self, X, padded):
        return self.lda.predict_proba(self.csp.logvar(_band_view(X, cfg.CSP_BAND, padded)))[:, 1]


# Filter-bank CSP: CSP per sub-band, mutual-information feature selection, then LDA
# Reference: Ang et al. (2008)
class FBCSPLDA:
    needs_source = False
    uses_covariances = False
    csp_fit_data = None  # audit only, as in CSPLDA

    def __init__(self, seed=0, n_components=4, k_features=8):
        self.seed, self.n_components, self.k_features = seed, n_components, k_features

    def _features(self, X):
        return np.hstack([c.logvar(_band_view(X, b, True)) for b, c in zip(cfg.FILTER_BANK, self.csps)])

    def fit(self, X, y, padded, source=None):
        if not padded:
            raise ValueError("FBCSP needs the raw recordings; it is undefined on P4 features")
        Xc, yc = self.csp_fit_data or (X, y)
        self.csps = [CSP(self.n_components).fit(_band_view(Xc, b, True), yc) for b in cfg.FILTER_BANK]
        F = self._features(X)
        selector = SelectKBest(partial(mutual_info_classif, random_state=self.seed),
                               k=min(self.k_features, F.shape[1]))
        self.clf = make_pipeline(selector, _lda()).fit(F, y)
        return self

    def predict_proba(self, X, padded):
        return self.clf.predict_proba(self._features(X))[:, 1]


BASELINES = {"csp_lda": CSPLDA, "fbcsp_lda": FBCSPLDA}


def _covariances(X, padded):
    from pyriemann.estimation import Covariances
    Z = _band_view(X, cfg.CSP_BAND, padded)
    return Covariances("oas").transform(Z.transpose(0, 2, 1).astype(np.float64))


# Trial covariances do not depend on the fold, so they are computed once per subject
def subject_covariances(subj, channels):
    from load_data import load_subject
    path = cfg.CACHE_DIR / f"covs_{subj}_{channels}.npy"
    if path.exists():
        return np.load(path)
    d = load_subject(subj, channels)
    C = _covariances(d["X"], d["padded"])
    cfg.CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, C)
    return C


def _recenter(C, M):
    from pyriemann.utils.base import invsqrtm
    P = invsqrtm(M)
    return P @ C @ P


def _identity_tangent(C):
    from pyriemann.utils.tangentspace import tangent_space
    return tangent_space(C, np.eye(C.shape[-1]), metric="riemann")


def _logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=2000)


# Riemannian tangent space at the training-set mean + logistic regression,
# on 8-30 Hz OAS covariances
# Reference: Barachant et al. (2012)
class TSLR:
    needs_source = False
    uses_covariances = True

    def __init__(self, seed=0):
        self.seed = seed

    def fit_cov(self, C, y, source=None):
        from pyriemann.tangentspace import TangentSpace
        self.clf = make_pipeline(TangentSpace(metric="riemann"), _logistic()).fit(C, y)
        return self

    def predict_proba_cov(self, C):
        return self.clf.predict_proba(C)[:, 1]


# Cross-subject transfer: every subject's covariances are re-centred at identity using its own
# Riemannian mean (target: training trials only), then source and target trials are pooled in
# the tangent space at identity.
# Reference: Zanini et al. (2018)
class XSTSLR(TSLR):
    needs_source = True

    def fit_cov(self, C, y, source):
        from pyriemann.utils.mean import mean_riemann
        self.M_ = mean_riemann(C)
        Fs, ys = source
        self.clf = _logistic().fit(np.vstack([Fs, _identity_tangent(_recenter(C, self.M_))]),
                                   np.concatenate([ys, y]))
        return self

    def predict_proba_cov(self, C):
        return self.clf.predict_proba(_identity_tangent(_recenter(C, self.M_)))[:, 1]


# Re-centred tangent features of every source subject (all trials; the target is never included)
def source_covariances(target, channels):
    from load_data import load_subject
    from pyriemann.utils.mean import mean_riemann
    path = cfg.CACHE_DIR / f"xs_tangent_{target}_{channels}.npz"
    if path.exists():
        d = np.load(path)
        return d["F"], d["y"]
    Fs, ys = [], []
    for s in cfg.SUBJECTS:
        if s == target:
            continue
        C = subject_covariances(s, channels)
        Fs.append(_identity_tangent(_recenter(C, mean_riemann(C))))
        ys.append(load_subject(s, channels)["y"])
    F, y = np.vstack(Fs), np.concatenate(ys)
    cfg.CACHE_DIR.mkdir(exist_ok=True)
    np.savez(path, F=F, y=y)
    return F, y


BASELINES.update({"ts_lr": TSLR, "xs_ts_lr": XSTSLR})
