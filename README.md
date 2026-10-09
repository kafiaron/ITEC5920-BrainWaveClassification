# How much of reported motor imagery accuracy is protocol? A leakage audit on BCI Competition III IVa

Two parts:
1. Protocol audit. Common evaluation errors are reproduced one at a time (--leak) and their effect
   is measured against the clean pipeline on identical folds and seeds.
2. Leak-free benchmark. Within-subject and cross-subject decoders, classical and deep, across
   calibration-set sizes and electrode montages.

## Data
Place the data next to the code:

    Raw BCI Data/data_set_IVa_{aa,al,av,aw,ay}.mat   MATLAB 100 Hz version,
                                                     https://www.bbci.de/competition/iii/desc_IVa.html
    P4/true_labels_{aa,al,av,aw,ay}.mat              true labels of the test trials (released after the competition)
    P4/features_{subj}_440_0.mat                     optional, legacy 6-dim features for --channels p4

The official test set is read from the raw files (trials without public labels).

## Setup on a GPU (Linux or WSL2)
TensorFlow only supports NVIDIA GPUs on Linux. On Windows, use WSL2 (Ubuntu), since native
Windows has no TensorFlow GPU support after version 2.10.

    python3 -m venv ~/venvs/eeg
    source ~/venvs/eeg/bin/activate
    pip install -r requirements.txt
    pip install "tensorflow[and-cuda]"
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

The last line must list at least one GPU. If it prints [], check the NVIDIA driver with
nvidia-smi (in WSL2, install the Windows NVIDIA driver; do not install a Linux driver inside WSL).

CPU-only machines need only: pip install -r requirements.txt

## Running
Smoke test (about a minute):

    python main.py --model ctnet --subjects al --channels motor8 --quick --max-folds 1 --align

Full experiments, 4 jobs at a time sharing the GPU:

    bash run_parallel.sh experiments_tier1.sh 4    # core results
    bash run_parallel.sh experiments_tier2.sh 4    # ablations

Watch progress with tail -f logs/job_*.log and nvidia-smi -l 5. Lower the job count if GPU
memory runs out, raise it if utilisation stays low. Every job is resumable: re-running skips
finished folds, and pretrained weights are cached in cache/pretrained.

--quick and --max-folds are for testing only; never report numbers produced with them.

## Methods (--model)
    csp_lda     CSP log-variance + shrinkage LDA
    fbcsp_lda   filter-bank CSP + mutual-information selection + LDA (Ang et al. 2008)
    ts_lr       Riemannian tangent space + logistic regression (Barachant et al. 2012)
    xs_ts_lr    cross-subject: per-subject Riemannian re-centring, pooled with the other four
                subjects (Zanini et al. 2018)
    hybrid      proposed dual-aligned hybrid: CTNet branch on Euclidean-aligned EEG plus a linear
                branch on tangent features of Riemannian re-centred covariances, summed logits;
                --pretrain pretrains the whole model on the other subjects
    eegnet, ctnet, tcformer, siamese   deep models; deviations in models.ARCHITECTURE_NOTES

Deep-model options: --align (Euclidean alignment), --pretrain (leave-one-subject-out pretraining),
--aug none|replicate|noise|scale|mask|all, --csp N, --bag K, --channels all|motor33|motor16|motor8|p4.

## Protocol audit (--leak, audit only)
    es_test      early stopping and model selection on the test fold
    tune_global  hyperparameters chosen once on a 75/25 split of all trials, before CV
    norm_global  alignment and z-scoring fit on all trials, including the test fold
    csp_global   CSP filters fit on all trials and labels, including the test fold
Several can be combined. Leaky runs skip the leakage assertions and are labelled +leak:<name>.
Reports:
    python summarize.py --protocol cv --audit
    python summarize.py --protocol cv --bestof "<method A>" "<method B>" ...
--bestof measures the effect of picking, per subject, the condition with the best test score.

## Evaluation protocol
Per subject, repeat and outer fold:
1. Outer split: StratifiedKFold(5, seed 42 + repeat) (--protocol cv), the official competition
   split (--protocol official), or subsampled training sets (--protocol curve --n-train ...).
2. Outer-train -> FIT 80% / VAL 20%, stratified. Below 40 trials there is no VAL split:
   fixed epochs and configuration 0.
3. Euclidean alignment and z-scoring are fit on outer-train; CSP on un-augmented FIT.
4. Hyperparameters (3 configurations per deep model) are tuned on un-augmented FIT/VAL and held
   fixed across augmentation conditions. Selection: VAL accuracy, then VAL log-loss, then grid order.
5. The final model is trained on (augmented) FIT and early-stopped on VAL. With --bag K, K models
   are trained on inner K-fold splits of outer-train and averaged.
6. TEST is predicted once. Disjoint FIT/VAL/TEST and augmentation provenance are asserted every fold.

Pretraining and cross-subject methods use the other four subjects only.
Seeds: fold_seed = 42 + 10000*subject + 100*repeat + fold, shared by every method and condition.

## Outputs (results/)
    folds.csv          one row per fold: all metrics, chosen config, set sizes, epochs, params, time
    tuning.csv         validation score of every configuration in every fold
    predictions.csv    test-trial probabilities
    manifests/*.json   every setting, grid, seed rule, software version and architecture note
    param_counts.csv, model_cards.txt   from: python main.py --describe

Metrics per test fold, then averaged: accuracy, balanced accuracy, binary F1 of class 1 (right
foot), macro F1, ROC-AUC from probabilities, Cohen's kappa. Decision threshold 0.5.

## Paper tables and figures
    python report.py
writes every table (Markdown and LaTeX), corrected CIs, the audit table and the learning-curve
figure (PDF and PNG) to results/report/.

## Statistics
    python summarize.py --protocol cv                 mean ± SD over repeats, per subject
    python summarize.py --protocol cv --ci            Nadeau & Bengio corrected 95% CI
    python summarize.py --protocol official
    python summarize.py --protocol curve              accuracy vs number of calibration trials
    python summarize.py --protocol cv --compare "ts_lr [all]" "ctnet [all] +EA +PT"
    python summarize.py --protocol cv --latex

--compare runs a per-subject corrected repeated k-fold t-test (Bouckaert & Frank 2004) and
Wilcoxon test, Holm-adjusted across subjects, plus group-level counts. With 5 subjects, a two-sided
Wilcoxon test across subjects cannot go below p = 0.0625, so group results are reported descriptively.
