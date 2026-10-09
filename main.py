# Run ONE model under one protocol. Rows append to results/*.csv; finished folds are skipped
# on rerun, so a crashed or interrupted job resumes where it stopped.
#
# Protocol per subject, repeat and outer fold:
#   1. Outer split: StratifiedKFold(5, seed = 42 + repeat), or the official competition split
#   2. Outer-train -> FIT 80% / VAL 20% (stratified); no VAL below 40 trials
#   3. Euclidean alignment and z-scoring fit on outer-train; CSP fit on FIT (un-augmented)
#   4. Hyperparameters tuned on un-augmented FIT/VAL, then held fixed across aug conditions
#   5. Final model trained on (augmented) FIT, early-stopped on VAL
#   6. TEST predicted once
#
# Examples:
#   python main.py --model fbcsp_lda --protocol cv --repeats 0 1 2 3 4
#   python main.py --model ctnet --align --pretrain --protocol cv --repeats 0 1 2
#   python main.py --model ctnet --align --aug noise --protocol cv --repeats 0 1 2
#   python main.py --model ctnet --align --pretrain --protocol curve --n-train 10 20 40 80 160
#   python main.py --model siamese --subjects al --channels motor8 --quick --max-folds 1   # smoke test
#   python main.py --describe                                                         # param counts
import argparse
import csv
import json
import os
import platform
import re
import time

try:
    import fcntl
except ImportError:
    fcntl = None

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import scipy
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import config as cfg
from augmentation import AUG_CONDITIONS, augment_with_provenance
from baselines import BASELINES
from load_data import crop, load_subject
from preprocessing import CSP, EuclideanAligner, Standardizer

DEEP_MODELS = ["eegnet", "ctnet", "tcformer", "siamese", "hybrid"]

# Protocol errors that can be reproduced on purpose, so the audit measures each one in isolation:
#   es_test      early stopping and model selection on the test fold (inner split dropped)
#   tune_global  hyperparameters chosen once on a 75/25 split of all trials, before CV
#   norm_global  alignment and z-scoring fit on all trials, including the test fold
#   csp_global   CSP filters fit on all trials and labels, including the test fold
LEAKS = ["es_test", "tune_global", "norm_global", "csp_global"]
KEY = ["tag", "method", "subject", "protocol", "repeat", "fold", "n_train"]
FOLD_FIELDS = KEY + ["model", "channels", "aug", "align", "pretrain", "csp", "fold_seed",
                     "hp_index", "n_fit", "n_val", "n_test", "n_params", "best_epoch",
                     "epochs_run", "acc", "bal_acc", "f1_pos", "f1_macro", "auc", "kappa",
                     "seconds", "timestamp"]
TUNING_FIELDS = KEY + ["hp_index", "hp", "val_acc", "val_logloss", "selected"]
PRED_FIELDS = KEY + ["trial", "y_true", "p_foot"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(BASELINES) + DEEP_MODELS)
    p.add_argument("--subjects", nargs="+", default=cfg.SUBJECTS)
    p.add_argument("--protocol", choices=["cv", "official", "curve"], default="cv")
    p.add_argument("--repeats", nargs="+", type=int, default=[0])
    p.add_argument("--channels", default="all", choices=["all", "p4"] + list(cfg.CHANNEL_SETS))
    p.add_argument("--aug", default="none", choices=AUG_CONDITIONS)
    p.add_argument("--align", action="store_true", help="Euclidean alignment")
    p.add_argument("--pretrain", action="store_true", help="leave-one-subject-out pretraining")
    p.add_argument("--csp", type=int, default=0, help="feed N CSP components to the deep model")
    p.add_argument("--no-tune", action="store_true", help="skip the nested hyperparameter search")
    p.add_argument("--leak", nargs="+", default=[], choices=LEAKS,
                   help="AUDIT ONLY: deliberately reproduce a protocol error to measure its effect")
    p.add_argument("--bag", type=int, default=1,
                   help="K>1: after tuning, train K models on inner K-fold splits of outer-train "
                        "(each early-stopped on its own held-out part) and average them")
    p.add_argument("--n-train", nargs="+", type=int, default=[10, 20, 40, 80, 160])
    p.add_argument("--quick", action="store_true", help="few epochs, no tuning: testing only")
    p.add_argument("--max-folds", type=int, default=None, help="limit folds: testing only")
    p.add_argument("--tag", default="main")
    p.add_argument("--describe", action="store_true", help="write parameter counts and exit")
    args = p.parse_args()

    if args.describe:
        return args
    if args.model is None:
        p.error("--model is required")
    if args.model in BASELINES and (args.pretrain or args.align or args.csp or args.aug != "none"):
        p.error("--align/--pretrain/--csp/--aug apply to deep models only")
    if args.bag > 1 and args.csp:
        p.error("--bag cannot be combined with --csp (CSP would need refitting per bag)")
    deep_only = {"es_test", "tune_global", "norm_global"}
    if args.model in BASELINES and deep_only & set(args.leak):
        p.error(f"{sorted(deep_only & set(args.leak))} apply to deep models only")
    if "csp_global" in args.leak and args.model in DEEP_MODELS and not args.csp:
        p.error("--leak csp_global needs --csp for deep models")
    if "csp_global" in args.leak and args.model in ("ts_lr", "xs_ts_lr"):
        p.error("--leak csp_global applies to CSP-based models only")
    if args.model == "hybrid":
        if args.aug != "none" or args.csp or args.bag > 1 or args.leak:
            p.error("hybrid supports --pretrain, --no-tune and the protocols only")
        args.align = True  # both branches are always aligned
    if args.pretrain and args.csp:
        p.error("CSP filters are subject-specific, so --csp cannot be combined with --pretrain")
    if args.model == "fbcsp_lda" and args.channels == "p4":
        p.error("FBCSP needs raw recordings")
    if args.quick:
        args.no_tune = True
        if args.tag == "main":
            args.tag = "quick"
    return args


def method_name(args):
    parts = [args.model, f"[{args.channels}]"]
    if args.csp:
        parts.append(f"+CSP{args.csp}")
    if args.aug != "none":
        parts.append(f"+aug:{args.aug}")
    if args.align:
        parts.append("+EA")
    if args.pretrain:
        parts.append("+PT")
    if args.bag > 1:
        parts.append(f"+bag{args.bag}")
    for leak in sorted(args.leak):
        parts.append(f"+leak:{leak}")
    return " ".join(parts)


# Metrics per test fold, threshold 0.5 on the predicted probability of class 1 (right foot).
# f1_pos = binary F1 of class 1; auc from probabilities, not hard labels.
def compute_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
                f1_pos=f1_score(y, pred, zero_division=0),
                f1_macro=f1_score(y, pred, average="macro", zero_division=0),
                auc=roc_auc_score(y, p), kappa=cohen_kappa_score(y, pred))


def outer_splits(protocol, d, repeat):
    y = d["y"]
    if protocol == "official":
        test = d["test_idx"]
        return [(np.setdiff1d(np.arange(len(y)), test), test)]
    skf = StratifiedKFold(cfg.N_FOLDS, shuffle=True, random_state=cfg.BASE_SEED + repeat)
    return list(skf.split(y, y))


def subsample(idx, y, n, seed):
    if n is None or n > len(idx) - 2:
        return idx
    keep, _ = next(StratifiedShuffleSplit(1, train_size=n, random_state=seed).split(idx, y[idx]))
    return idx[keep]


def inner_split(y, seed):
    if len(y) < cfg.MIN_TRIALS_FOR_VAL:
        return np.arange(len(y)), None
    sss = StratifiedShuffleSplit(1, test_size=cfg.INNER_VAL_FRACTION, random_state=seed)
    return next(sss.split(y, y))


def assert_no_leakage(fit_ids, val_ids, test_ids, aug_src):
    fit_ids, test_ids = set(fit_ids), set(test_ids)
    val_ids = set(val_ids) if val_ids is not None else set()
    assert not fit_ids & val_ids, "FIT and VAL overlap"
    assert not fit_ids & test_ids, "FIT and TEST overlap"
    assert not val_ids & test_ids, "VAL and TEST overlap"
    assert set(aug_src) <= fit_ids, "augmented rows derived from non-FIT trials"


def source_data(target, channels, align):
    Xs, ys = [], []
    for s in cfg.SUBJECTS:
        if s == target:
            continue
        d = load_subject(s, channels)
        X = crop(d["X"], d["padded"])
        if align:
            X = EuclideanAligner().fit(X).transform(X)
        Xs.append(Standardizer().fit(X).transform(X))
        ys.append(d["y"])
    return np.concatenate(Xs), np.concatenate(ys)


# Tangent features of covariances re-centred by the Riemannian mean of the training trials,
# z-scored with training statistics
def tangent_features(subj, channels, tr, te):
    from baselines import _identity_tangent, _recenter, subject_covariances
    from pyriemann.utils.mean import mean_riemann
    C = subject_covariances(subj, channels)
    M = mean_riemann(C[tr])
    Ftr, Fte = (_identity_tangent(_recenter(C[i], M)) for i in (tr, te))
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    return ((Ftr - mu) / sd).astype(np.float32), ((Fte - mu) / sd).astype(np.float32)


def source_hybrid(target, channels):
    from baselines import source_covariances
    Xs, ys = source_data(target, channels, align=True)
    Fs, yf = source_covariances(target, channels)
    assert np.array_equal(ys, yf), "source raw and covariance data out of order"
    Fs = ((Fs - Fs.mean(0)) / (Fs.std(0) + 1e-8)).astype(np.float32)
    return [Xs, Fs], ys


def run_hybrid(args, d, tr, te, seed, tc, subj, source_cache, repeat):
    from training import Net, pretrain, set_seed, validation_score

    t0 = time.time()
    X, y = crop(d["X"], d["padded"]), d["y"]
    ea = EuclideanAligner().fit(X[tr])
    Xtr, Xte = ea.transform(X[tr]), ea.transform(X[te])
    sc = Standardizer().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Ftr, Fte = tangent_features(subj, args.channels, tr, te)

    ytr = y[tr]
    fit, val = inner_split(ytr, seed)
    assert_no_leakage(tr[fit], tr[val] if val is not None else None, te, tr[fit])
    Xf, yf = [Xtr[fit], Ftr[fit]], ytr[fit]
    validation = ([Xtr[val], Ftr[val]], ytr[val]) if val is not None else None

    def train(hi, hp):
        hp = dict(hp, n_tangent=Ftr.shape[1])
        lr, weights = tc.lr, None
        if args.pretrain:
            if subj not in source_cache:
                source_cache.clear()
                source_cache[subj] = source_hybrid(subj, args.channels)
            pt_seed = cfg.fold_seed(subj, repeat, 0)
            weights = (cfg.CACHE_DIR / "pretrained" /
                       f"hybrid_hp{hi}_{subj}_{args.channels}_s{pt_seed}_{tc.tag}.weights.h5")
            pretrain("hybrid", hp, *source_cache[subj], tc, pt_seed, weights)
            lr = tc.lr_finetune
        set_seed(seed)
        net = Net("hybrid", Xtr.shape[1], Xtr.shape[2], hp)
        if weights is not None:
            net.load(weights)
        info = net.fit(Xf, yf, np.arange(len(yf)), validation, tc, lr, np.random.default_rng(seed))
        return net, info

    grid = cfg.HP_GRIDS["hybrid"]
    tuning_rows, best_hi = [], 0
    if validation is not None and not args.no_tune:
        best_score = None
        for hi, hp in enumerate(grid):
            cand, cand_info = train(hi, hp)
            score = validation_score(cand, *validation)
            tuning_rows.append(dict(hp_index=hi, hp=json.dumps(hp), val_acc=round(score[0], 4),
                                    val_logloss=round(-score[1], 4), selected=0))
            if best_score is None or score > best_score:
                best_score, best_hi, net, info = score, hi, cand, cand_info
        tuning_rows[best_hi]["selected"] = 1
    else:
        net, info = train(0, grid[0])

    p = net.predict([Xte, Fte])
    meta = dict(hp_index=best_hi, n_fit=len(yf), n_val=0 if val is None else len(val),
                n_params=net.count_params(), **info)
    return p, meta, tuning_rows, t0


def run_baseline(args, d, tr, te, seed, subj):
    from baselines import source_covariances, subject_covariances
    t0 = time.time()
    cls = BASELINES[args.model]
    source = source_covariances(subj, args.channels) if cls.needs_source else None
    if cls.uses_covariances:
        C = subject_covariances(subj, args.channels)
        clf = cls(seed=seed).fit_cov(C[tr], d["y"][tr], source)
        p = clf.predict_proba_cov(C[te])
    else:
        clf = cls(seed=seed)
        if "csp_global" in args.leak:
            clf.csp_fit_data = (d["X"], d["y"])
        clf.fit(d["X"][tr], d["y"][tr], d["padded"], source)
        p = clf.predict_proba(d["X"][te], d["padded"])
    return p, dict(hp_index=0, n_fit=len(tr), n_val=0, n_params="", best_epoch="", epochs_run=""), [], t0


def run_deep(args, d, tr, te, seed, tc, subj, source_cache, repeat, forced_hp=None):
    from training import Net, pretrain, set_seed, validation_score

    t0 = time.time()
    X, y = crop(d["X"], d["padded"]), d["y"]
    Xtr, ytr, Xte = X[tr], y[tr], X[te]

    norm_data = X if "norm_global" in args.leak else Xtr
    if args.align:
        ea = EuclideanAligner().fit(norm_data)
        Xtr, Xte = ea.transform(Xtr), ea.transform(Xte)
        norm_data = ea.transform(norm_data)
    sc = Standardizer().fit(norm_data)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    if "es_test" in args.leak:
        fit, val = np.arange(len(tr)), None
        Xf, yf, src_f = Xtr, ytr, tr
        Xv, yv = Xte, y[te]
    else:
        fit, val = inner_split(ytr, seed)
        Xf, yf, src_f = Xtr[fit], ytr[fit], tr[fit]
        Xv, yv = (Xtr[val], ytr[val]) if val is not None else (None, None)

    if args.csp:
        if "csp_global" in args.leak:
            csp = CSP(args.csp).fit(np.concatenate([Xtr, Xte]), np.concatenate([ytr, y[te]]))
        else:
            csp = CSP(args.csp).fit(Xf, yf)
        Xf, Xte = csp.project(Xf), csp.project(Xte)
        Xv = csp.project(Xv) if Xv is not None else None
        sc2 = Standardizer().fit(Xf)
        Xf, Xte = sc2.transform(Xf), sc2.transform(Xte)
        Xv = sc2.transform(Xv) if Xv is not None else None

    Xa, ya, src_a = augment_with_provenance(Xf, yf, src_f, args.aug, np.random.default_rng(seed + 7))
    if not args.leak:
        assert_no_leakage(src_f, tr[val] if val is not None else None, te, src_a)
    validation = (Xv, yv) if Xv is not None else None

    def train(hi, hp, Xt, yt, st, val_data=None, run_seed=None):
        val_data = validation if val_data is None else val_data
        run_seed = seed if run_seed is None else run_seed
        lr, weights = tc.lr, None
        if args.pretrain:
            if subj not in source_cache:
                source_cache.clear()
                source_cache[subj] = source_data(subj, args.channels, args.align)
            # Source data never contains the target, so one pretrained model per
            # (target, config, repeat) is shared by every fold, protocol and condition
            pt_seed = cfg.fold_seed(subj, repeat, 0)
            weights = (cfg.CACHE_DIR / "pretrained" /
                       f"{args.model}_hp{hi}_{subj}_{args.channels}_ea{int(args.align)}"
                       f"_s{pt_seed}_{tc.tag}.weights.h5")
            pretrain(args.model, hp, *source_cache[subj], tc, pt_seed, weights)
            lr = tc.lr_finetune
        set_seed(run_seed)
        net = Net(args.model, Xt.shape[1], Xt.shape[2], hp)
        if weights is not None:
            net.load(weights)
        info = net.fit(Xt, yt, st, val_data, tc, lr, np.random.default_rng(run_seed))
        return net, info

    grid = cfg.HP_GRIDS[args.model]
    tuning_rows, net, info, best_hi = [], None, None, 0
    if forced_hp is not None:
        net, info = train(forced_hp, grid[forced_hp], Xa, ya, src_a)
        best_hi = forced_hp
    elif validation is not None and not args.no_tune:
        best_score = None
        for hi, hp in enumerate(grid):
            cand, cand_info = train(hi, hp, Xf, yf, src_f)
            score = validation_score(cand, Xv, yv)
            tuning_rows.append(dict(hp_index=hi, hp=json.dumps(hp), val_acc=round(score[0], 4),
                                    val_logloss=round(-score[1], 4), selected=0))
            if best_score is None or score > best_score:
                best_score, best_hi, net, info = score, hi, cand, cand_info
        tuning_rows[best_hi]["selected"] = 1
        if args.aug != "none":
            net, info = train(best_hi, grid[best_hi], Xa, ya, src_a)
    else:
        net, info = train(0, grid[0], Xa, ya, src_a)

    p = net.predict(Xte)
    n_fit, n_val = len(ya), 0 if val is None else len(val)

    # Bagging over inner folds: every outer-training trial updates some model's weights
    # while each model still early-stops on data it never trained on
    if args.bag > 1 and validation is not None and "es_test" not in args.leak:
        preds, infos = [], []
        skf = StratifiedKFold(args.bag, shuffle=True, random_state=seed)
        for k, (bf, bv) in enumerate(skf.split(ytr, ytr)):
            Xb, yb, sb = augment_with_provenance(Xtr[bf], ytr[bf], tr[bf], args.aug,
                                                 np.random.default_rng(seed + 7 + k))
            assert_no_leakage(tr[bf], tr[bv], te, sb)
            bag_net, bag_info = train(best_hi, grid[best_hi], Xb, yb, sb,
                                      (Xtr[bv], ytr[bv]), seed + 1000 * (k + 1))
            preds.append(bag_net.predict(Xte))
            infos.append(bag_info)
        p = np.mean(preds, axis=0)
        info = dict(best_epoch=round(np.mean([i["best_epoch"] for i in infos]), 1),
                    epochs_run=round(np.mean([i["epochs_run"] for i in infos]), 1))
        n_fit, n_val = len(Xb), len(bv)

    meta = dict(hp_index=best_hi, n_fit=n_fit, n_val=n_val, n_params=net.count_params(), **info)
    return p, meta, tuning_rows, t0


# LEAK (audit only): reproduces the submitted Siamese protocol, where hyperparameters were
# chosen once on a 75/25 split of all 280 trials whose validation part later became CV test data
def global_tune(args, d, tc, subj, source_cache, repeat, cache):
    key = (subj, repeat)
    if key not in cache:
        y = d["y"]
        sss = StratifiedShuffleSplit(1, test_size=0.25, random_state=cfg.BASE_SEED + repeat)
        a, b = next(sss.split(y, y))
        seed = cfg.fold_seed(subj, repeat, 99)
        scores = []
        for hi in range(len(cfg.HP_GRIDS[args.model])):
            p, *_ = run_deep(args, d, a, b, seed, tc, subj, source_cache, repeat, forced_hp=hi)
            scores.append(accuracy_score(y[b], p >= 0.5))
        cache[key] = int(np.argmax(scores))
    return cache[key]


class Writer:
    def __init__(self, name, fields):
        self.path, self.fields = cfg.RESULTS_DIR / name, fields

    def keys(self):
        if not self.path.exists():
            return set()
        with open(self.path) as f:
            return {tuple(r[k] for k in KEY) for r in csv.DictReader(f)}

    # File lock so parallel jobs can append to the same CSV safely (Linux/WSL)
    def write(self, rows):
        with open(self.path, "a", newline="") as f:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=self.fields)
            if f.tell() == 0:
                w.writeheader()
            w.writerows(rows)
            f.flush()
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_UN)


def write_manifest(args, method, tc):
    from models import ARCHITECTURE_NOTES
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    sklearn=sklearn.__version__)
    if tc is not None:
        import tensorflow as tf
        versions["tensorflow"] = tf.__version__
    manifest = dict(
        method=method, args={k: v for k, v in vars(args).items()}, versions=versions,
        data=dict(fs=cfg.FS, epoch_window=cfg.EPOCH_WINDOW, crop_window=cfg.CROP_WINDOW,
                  broad_band=cfg.BROAD_BAND, csp_band=cfg.CSP_BAND, filter_bank=cfg.FILTER_BANK,
                  channel_sets=cfg.CHANNEL_SETS),
        protocol=dict(n_folds=cfg.N_FOLDS, inner_val_fraction=cfg.INNER_VAL_FRACTION,
                      min_trials_for_val=cfg.MIN_TRIALS_FOR_VAL, outer_seed="42 + repeat",
                      fold_seed="42 + 10000*subject_index + 100*repeat + fold",
                      selection="val accuracy, then val log-loss, then grid order",
                      tuning="on un-augmented FIT, held fixed across augmentation conditions",
                      threshold=0.5),
        augmentation=dict(params=cfg.AUG_PARAMS, copies=cfg.AUG_COPIES),
        hp_grid=cfg.HP_GRIDS.get(args.model),
        training=tc.as_dict() if tc is not None else None,
        architecture=ARCHITECTURE_NOTES.get(args.model) if tc is not None else None,
    )
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{args.tag}_{method}_{args.protocol}").strip("_")
    path = cfg.RESULTS_DIR / "manifests" / f"{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))


def describe(args):
    import tensorflow as tf
    from training import Net
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    rows, cards = [], []
    for channels in ["all"] + list(cfg.CHANNEL_SETS):
        d = load_subject(cfg.SUBJECTS[0], channels)
        T, C = crop(d["X"], d["padded"]).shape[1:]
        for name in DEEP_MODELS:
            for hi, hp in enumerate(cfg.HP_GRIDS[name]):
                tf.keras.backend.clear_session()
                net = Net(name, T, C, dict(hp, n_tangent=C * (C + 1) // 2) if name == "hybrid" else hp)
                rows.append(dict(model=name, channels=channels, C=C, T=T, hp_index=hi,
                                 hp=json.dumps(hp), params=net.count_params()))
                if channels == "all" and hi == 0:
                    lines = []
                    net.classifier.summary(print_fn=lambda s, **_: lines.append(s))
                    cards.append(f"=== {name} (hp0, {C} channels) ===\n" + "\n".join(lines))
    with open(cfg.RESULTS_DIR / "param_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (cfg.RESULTS_DIR / "model_cards.txt").write_text("\n\n".join(cards))
    for r in rows:
        if r["channels"] == "all":
            print(f"{r['model']:<9} hp{r['hp_index']} params={r['params']:,}")
    print("Wrote results/param_counts.csv and results/model_cards.txt")


def main():
    args = parse_args()
    if args.describe:
        describe(args)
        return

    tc = None
    if args.model in DEEP_MODELS:
        from training import TrainConfig
        tc = TrainConfig.quick() if args.quick else TrainConfig()

    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    folds_out = Writer("folds.csv", FOLD_FIELDS)
    tuning_out = Writer("tuning.csv", TUNING_FIELDS)
    preds_out = Writer("predictions.csv", PRED_FIELDS)
    done = folds_out.keys()
    method = method_name(args)
    write_manifest(args, method, tc)
    n_trains = args.n_train if args.protocol == "curve" else [None]
    source_cache, global_cache = {}, {}

    for repeat in args.repeats:
        for subj in args.subjects:
            d = load_subject(subj, args.channels)
            splits = outer_splits(args.protocol, d, repeat)[:args.max_folds]
            for n_train in n_trains:
                for fold, (tr, te) in enumerate(splits):
                    key_vals = dict(tag=args.tag, method=method, subject=subj, protocol=args.protocol,
                                    repeat=repeat, fold=fold, n_train=n_train or "")
                    if tuple(str(key_vals[k]) for k in KEY) in done:
                        continue
                    seed = cfg.fold_seed(subj, repeat, fold)
                    tr = subsample(tr, d["y"], n_train, seed)
                    if args.model in BASELINES:
                        p, meta, tuning_rows, t0 = run_baseline(args, d, tr, te, seed, subj)
                    elif args.model == "hybrid":
                        p, meta, tuning_rows, t0 = run_hybrid(args, d, tr, te, seed, tc, subj,
                                                              source_cache, repeat)
                    else:
                        forced = None
                        if "tune_global" in args.leak:
                            forced = global_tune(args, d, tc, subj, source_cache, repeat, global_cache)
                        p, meta, tuning_rows, t0 = run_deep(args, d, tr, te, seed, tc, subj, source_cache,
                                                            repeat, forced)
                    m = compute_metrics(d["y"][te], p)
                    row = dict(**key_vals, model=args.model, channels=args.channels, aug=args.aug,
                               align=int(args.align), pretrain=int(args.pretrain), csp=args.csp,
                               fold_seed=seed, n_test=len(te), **meta,
                               **{k: round(v, 4) for k, v in m.items()},
                               seconds=round(time.time() - t0, 1),
                               timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
                    tuning_out.write([dict(**key_vals, **r) for r in tuning_rows])
                    preds_out.write([dict(**key_vals, trial=int(t), y_true=int(yt), p_foot=round(float(pp), 5))
                                     for t, yt, pp in zip(te, d["y"][te], p)])
                    folds_out.write([row])
                    print(f"{method} | {subj} | rep {repeat} | fold {fold} | n_train {len(tr)} | "
                          f"acc {m['acc']:.3f} auc {m['auc']:.3f} | {row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
