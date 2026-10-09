#!/usr/bin/env bash
# Core results. One command per line, each independent and resumable.
# Sequential:  bash experiments_tier1.sh
# Parallel:    bash run_parallel.sh experiments_tier1.sh 4

# Classical baselines (CPU, minutes)
python main.py --model csp_lda --protocol cv --repeats 0 1 2
python main.py --model fbcsp_lda --protocol cv --repeats 0 1 2
python main.py --model ts_lr --protocol cv --repeats 0 1 2
python main.py --model xs_ts_lr --protocol cv --repeats 0 1 2
python main.py --model csp_lda --protocol official
python main.py --model fbcsp_lda --protocol official
python main.py --model ts_lr --protocol official
python main.py --model xs_ts_lr --protocol official
python main.py --model fbcsp_lda --protocol curve --repeats 0 1 2 --n-train 10 20 40 80
python main.py --model ts_lr --protocol curve --repeats 0 1 2 --n-train 10 20 40 80
python main.py --model xs_ts_lr --protocol curve --repeats 0 1 2 --n-train 10 20 40 80

# Protocol audit: each reproduced error versus its clean counterpart on identical folds
python main.py --model csp_lda --leak csp_global --protocol cv --repeats 0 1 2
python main.py --model fbcsp_lda --leak csp_global --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --leak es_test --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --leak tune_global --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --leak norm_global --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --leak es_test tune_global norm_global --protocol cv --repeats 0 1 2
python main.py --model ctnet --csp 8 --protocol cv --repeats 0 1 2
python main.py --model ctnet --csp 8 --leak csp_global --protocol cv --repeats 0 1 2

# Input representation: legacy 6-dim P4 features vs full montage
python main.py --model ctnet --channels p4 --align --protocol cv --repeats 0 1 2
python main.py --model ctnet --channels all --align --protocol cv --repeats 0 1 2

# Proposed model: dual-aligned hybrid (run these first)
python main.py --model hybrid --protocol cv --repeats 0 1 2
python main.py --model hybrid --pretrain --protocol cv --repeats 0 1 2
python main.py --model hybrid --pretrain --protocol official --repeats 0 1 2
python main.py --model hybrid --pretrain --protocol curve --repeats 0 1 2 --n-train 10 20 40 80

# Deep models: within-subject vs cross-subject pretraining
python main.py --model eegnet --align --protocol cv --repeats 0 1 2
python main.py --model eegnet --align --pretrain --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --pretrain --protocol cv --repeats 0 1 2
python main.py --model eegnet --align --pretrain --protocol official --repeats 0 1 2
python main.py --model ctnet --align --pretrain --protocol official --repeats 0 1 2

# Calibration-size learning curves for the deep models
python main.py --model eegnet --align --pretrain --protocol curve --repeats 0 1 2 --n-train 10 20 40 80
python main.py --model ctnet --align --pretrain --protocol curve --repeats 0 1 2 --n-train 10 20 40 80
