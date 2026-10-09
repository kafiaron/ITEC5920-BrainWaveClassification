from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "Raw BCI Data"
P4_DIR = ROOT / "P4"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

SUBJECTS = ["aa", "al", "av", "aw", "ay"]
FS = 100

# Epochs are extracted with 0.5 s margins so band-pass filtering has no edge effects,
# then cropped to the analysis window.
EPOCH_WINDOW = (0.0, 3.0)
CROP_WINDOW = (0.5, 2.5)
BROAD_BAND = (4.0, 40.0)
CSP_BAND = (8.0, 30.0)
FILTER_BANK = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 40)]

CHANNEL_SETS = {
    "motor8": ["C3", "C1", "Cz", "C2", "C4", "CP3", "CPz", "CP4"],
    "motor16": ["FC3", "FCz", "FC4", "CFC3", "CFC1", "CFC2", "CFC4",
                "C3", "C1", "Cz", "C2", "C4", "CCP3", "CCP1", "CCP2", "CCP4"],
    "motor33": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                "CFC5", "CFC3", "CFC1", "CFC2", "CFC4", "CFC6",
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CCP5", "CCP3", "CCP1", "CCP2", "CCP4", "CCP6",
                "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
}

N_FOLDS = 5
INNER_VAL_FRACTION = 0.2
MIN_TRIALS_FOR_VAL = 40  # below this, no inner validation: fixed epochs, no tuning

# Equal tuning budget for every deep model (3 configs each).
_TRANSFORMER_GRID = [
    dict(F1=8, D=2, F2=16, heads=2, dropout=0.5),
    dict(F1=8, D=2, F2=16, heads=2, dropout=0.25),
    dict(F1=16, D=2, F2=32, heads=4, dropout=0.5),
]
HP_GRIDS = {
    "eegnet": [
        dict(F1=8, D=2, F2=16, dropout=0.5),
        dict(F1=8, D=2, F2=16, dropout=0.25),
        dict(F1=16, D=2, F2=32, dropout=0.5),
    ],
    "ctnet": list(_TRANSFORMER_GRID),
    "tcformer": list(_TRANSFORMER_GRID),
    "siamese": list(_TRANSFORMER_GRID),
    "hybrid": list(_TRANSFORMER_GRID),
}

# Seeds: outer CV partition = BASE_SEED + repeat; weights/dropout/augmentation = fold_seed.
# The same fold_seed is shared by every model and condition (common random numbers),
# which reduces noise in paired comparisons.
BASE_SEED = 42


def fold_seed(subject, repeat, fold):
    return BASE_SEED + 10000 * SUBJECTS.index(subject) + 100 * repeat + fold


AUG_PARAMS = dict(noise_rel_std=0.05, scale_range=(0.8, 1.2), mask_width=10)
AUG_COPIES = 3  # every augmented condition is original + 3 copies (4x), so sizes match
