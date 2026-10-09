#!/usr/bin/env bash
# Reviewer-requested ablations. Run after tier 1 if time allows.

# Bagging over inner folds
python main.py --model ctnet --align --pretrain --bag 5 --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --pretrain --bag 5 --protocol official --repeats 0 1 2

# Siamese and TCFormer
python main.py --model siamese --align --pretrain --protocol cv --repeats 0 1 2
python main.py --model tcformer --align --pretrain --protocol cv --repeats 0 1 2

# Augmentation, one transform at a time, plus the replicate control
python main.py --model ctnet --align --aug none --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --aug replicate --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --aug noise --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --aug scale --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --aug mask --protocol cv --repeats 0 1 2
python main.py --model ctnet --align --aug all --protocol cv --repeats 0 1 2

# CSP input on vs off (within-subject)
python main.py --model eegnet --csp 8 --protocol cv --repeats 0 1 2
python main.py --model eegnet --protocol cv --repeats 0 1 2
python main.py --model ctnet --csp 8 --protocol cv --repeats 0 1 2
python main.py --model ctnet --protocol cv --repeats 0 1 2

# Euclidean alignment on vs off under pretraining
python main.py --model ctnet --pretrain --protocol cv --repeats 0 1 2

# Reduced, consumer-style montages
python main.py --model fbcsp_lda --channels motor16 --protocol cv --repeats 0 1 2
python main.py --model ts_lr --channels motor16 --protocol cv --repeats 0 1 2
python main.py --model xs_ts_lr --channels motor16 --protocol cv --repeats 0 1 2
python main.py --model ts_lr --channels motor8 --protocol cv --repeats 0 1 2
python main.py --model xs_ts_lr --channels motor8 --protocol cv --repeats 0 1 2
python main.py --model ctnet --channels motor16 --align --pretrain --protocol cv --repeats 0 1 2
python main.py --model xs_ts_lr --channels motor16 --protocol curve --repeats 0 1 2 --n-train 10 20 40 80
python main.py --model ctnet --channels motor16 --align --pretrain --protocol curve --repeats 0 1 2 --n-train 10 20 40 80
