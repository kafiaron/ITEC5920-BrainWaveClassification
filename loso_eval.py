import numpy as np
import matplotlib.pyplot as plt
import os
from load_data import load_subject
from main import build_eegnet, build_baseline_cnn, T, C
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs("results", exist_ok=True)

SUBJECTS = ["aa", "al", "av", "aw", "ay"]
es = EarlyStopping(patience=10, restore_best_weights=True)

# Leave-One-Subject-Out Evaluation
print("\n═ Leave-One-Subject-Out Evaluation:")
loso_results = {"baseline": [], "eegnet": []}

for test_subj in SUBJECTS:
    print(f"\n─── Test Subject: {test_subj} ───")

    X_train_list, y_train_list = [], []
    for subj in SUBJECTS:
        X_s, y_s = load_subject("P4", subj)
        mean_s = X_s.mean(axis=(0, 1), keepdims=True)
        std_s  = X_s.std(axis=(0, 1),  keepdims=True) + 1e-8
        X_s    = (X_s - mean_s) / std_s
        if set(np.unique(y_s)) == {1, 2}:
            y_s = (y_s == 2).astype(np.float32)
        if subj == test_subj:
            X_test_s, y_test_s = X_s, y_s
        else:
            X_train_list.append(X_s)
            y_train_list.append(y_s)

    X_train_s = np.concatenate(X_train_list)[..., np.newaxis]
    y_train_s = np.concatenate(y_train_list)
    X_test_s  = X_test_s[..., np.newaxis]

    # Baseline CNN
    cnn = build_baseline_cnn(X_train_s.shape[1:])
    cnn.fit(X_train_s, y_train_s, epochs=100, batch_size=32,
            validation_split=0.1, callbacks=[es], verbose=0)
    _, acc = cnn.evaluate(X_test_s, y_test_s, verbose=0)
    loso_results["baseline"].append(acc)
    print(f"Baseline CNN: {acc:.4f}")

    # EEGNet
    eeg = build_eegnet(T, C)
    eeg.fit(X_train_s, y_train_s, epochs=100, batch_size=32,
            validation_split=0.1, callbacks=[es], verbose=0)
    _, acc = eeg.evaluate(X_test_s, y_test_s, verbose=0)
    loso_results["eegnet"].append(acc)
    print(f"EEGNet:       {acc:.4f}")

print("\n═ LOSO Final Results:")
for name, accs in loso_results.items():
    print(f"{name}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    for subj, acc in zip(SUBJECTS, accs):
        print(f"  {subj}: {acc:.4f}")

# LOSO Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(SUBJECTS))
ax.bar(x - 0.2, loso_results["baseline"], 0.4, label="Baseline CNN", color="steelblue")
ax.bar(x + 0.2, loso_results["eegnet"],   0.4, label="EEGNet",       color="coral")
ax.axhline(np.mean(loso_results["baseline"]), color="steelblue", linestyle="--", alpha=0.5)
ax.axhline(np.mean(loso_results["eegnet"]),   color="coral",     linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([s.upper() for s in SUBJECTS])
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("Leave-One-Subject-Out: Baseline CNN vs EEGNet")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("results/loso_comparison.png")
plt.close()
print("Saved: loso_comparison.png")

print("\n Done.")