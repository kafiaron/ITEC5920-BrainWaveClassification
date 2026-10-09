import numpy as np

import config as cfg

# "replicate" = identical copies; controls for the extra gradient steps of a 4x training set
AUG_CONDITIONS = ("none", "replicate", "noise", "scale", "mask", "all")


# Add Gaussian noise scaled by each trial's std
# Reference: Khoyani et al. (2022)
def _noise(X, rng):
    std = X.std(axis=(1, 2), keepdims=True)
    return X + rng.normal(0, cfg.AUG_PARAMS["noise_rel_std"], X.shape).astype(np.float32) * std


# Scale amplitude to simulate variation across trials
def _scale(X, rng):
    return X * rng.uniform(*cfg.AUG_PARAMS["scale_range"], (len(X), 1, 1)).astype(np.float32)


# Zero out a short random time segment
# Reference: Park et al. (2019) SpecAugment
def _mask(X, rng):
    width = cfg.AUG_PARAMS["mask_width"]
    out = X.copy()
    for i, s in enumerate(rng.integers(0, X.shape[1] - width, len(X))):
        out[i, s:s + width] = 0
    return out


_TRANSFORMS = {"replicate": lambda X, rng: X.copy(), "noise": _noise, "scale": _scale, "mask": _mask}


# Original trials plus AUG_COPIES transformed copies. Returns the source-trial index of every
# row so leakage can be asserted and Siamese positives can exclude copies of the anchor.
def augment_with_provenance(X, y, src, condition, rng):
    if condition == "none":
        return X, y, src
    names = ["noise", "scale", "mask"] if condition == "all" else [condition] * cfg.AUG_COPIES
    Xa = np.concatenate([X] + [_TRANSFORMS[n](X, rng) for n in names])
    k = len(names) + 1
    perm = rng.permutation(len(X) * k)
    return Xa[perm].astype(np.float32), np.tile(y, k)[perm], np.tile(src, k)[perm]
