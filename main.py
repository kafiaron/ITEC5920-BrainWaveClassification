import numpy as np
import matplotlib.pyplot as plt
import os
from load_data import build_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, AveragePooling2D, Dropout,
    Flatten, Dense, Input, Activation
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

os.makedirs("results", exist_ok=True)

# Load and Normalize for each subject
from load_data import load_subject

SUBJECTS = ["aa", "al", "av", "aw", "ay"]
X_list, y_list = [], []

for subj in SUBJECTS:
    X_s, y_s = load_subject("P4", subj)
    mean_s = X_s.mean(axis=(0, 1), keepdims=True)
    std_s  = X_s.std(axis=(0, 1),  keepdims=True) + 1e-8
    X_s    = (X_s - mean_s) / std_s
    X_list.append(X_s)
    y_list.append(y_s)

X = np.concatenate(X_list)
y = np.concatenate(y_list)

# Convert labels from {1,2} to {0,1}
if set(np.unique(y)) == {1, 2}:
    y = (y == 2).astype(np.float32)

print("Dataset shape:", X.shape)   # (1400, 201, 6)
print("Labels shape:", y.shape)    # (1400,)
print("Unique labels:", np.unique(y, return_counts=True))

# Reshape for EEGNet: (trials, time, channels, 1)
X_eeg = X[..., np.newaxis]        # (1400, 201, 6, 1)
T = X_eeg.shape[1]                # 201 timepoints
C = X_eeg.shape[2]                # 6 channels

# EEGNet (Reference: Lawhern et al. 2018)
# Input: (batch, time, channels, 1)
# Block 1: temporal convolution (T//2, 1) then depthwise spatial convolution (1, C)
# Block 2: separable convolution (16, 1) for time summary + pointwise mixing
def build_eegnet(T, C, F1=8, D=2, F2=16, dropout=0.5):
    inputs = Input(shape=(T, C, 1))

    # Block 1 - Temporal convolution (learns frequency filters)
    x = Conv2D(F1, (T // 2, 1), padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    # Depthwise spatial convolution (learns spatial filters per frequency)
    x = DepthwiseConv2D((1, C), depth_multiplier=D,
                         depthwise_constraint=max_norm(1.),
                         use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((4, 1))(x)
    x = Dropout(dropout)(x)

    # Block 2 - Separable convolution (temporal summary + feature mixing)
    x = SeparableConv2D(F2, (16, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((8, 1))(x)
    x = Dropout(dropout)(x)

    # Classifier
    x = Flatten()(x)
    outputs = Dense(1, activation="sigmoid",
                    kernel_constraint=max_norm(0.25))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# Baseline CNN (for comparison)
# L2 regularization + reduced model size to fix overfitting
from tensorflow.keras.regularizers import l2

def build_baseline_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, (5, 1), padding="same", activation="relu",
               kernel_regularizer=l2(1e-4)),
        Conv2D(32, (5, 1), padding="same", activation="relu",
               kernel_regularizer=l2(1e-4)),
        AveragePooling2D((4, 1)),
        Dropout(0.5),                       
        Flatten(),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# Visualization: Raw EEG Signal
fig, axes = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
trial = X[0]   # shape (201, 6)
for i, ax in enumerate(axes):
    ax.plot(trial[:, i])
    ax.set_ylabel(f"Ch {i+1}")
axes[-1].set_xlabel("Timepoints")
plt.suptitle("Raw EEG Signal - One Trial")
plt.tight_layout()
plt.savefig("results/eeg_signal.png")
plt.close()
print("Saved: eeg_signal.png")

# Visualization: Spectrogram 
from scipy.signal import spectrogram as scipy_spectrogram
f, t, Sxx = scipy_spectrogram(X[0][:, 0], fs=100, nperseg=64)
plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.title("EEG Spectrogram - Channel 1, Trial 1")
plt.colorbar(label="Power (dB)")
plt.savefig("results/spectrogram.png")
plt.close()
print("Saved: spectrogram.png")

# 5-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
es = EarlyStopping(patience=10, restore_best_weights=True)
results = {"baseline": [], "eegnet": []}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_eeg, y)):
    print(f"\n─── Fold {fold+1}/5 ───")
    X_tr,  X_val  = X_eeg[train_idx], X_eeg[val_idx]
    y_tr,  y_val  = y[train_idx],     y[val_idx]

    # Baseline CNN
    cnn = build_baseline_cnn(X_tr.shape[1:])
    cnn.fit(X_tr, y_tr, epochs=100, batch_size=32,
            validation_split=0.1, callbacks=[es], verbose=0)
    _, acc = cnn.evaluate(X_val, y_val, verbose=0)
    results["baseline"].append(acc)
    print(f"Baseline CNN - Fold {fold+1}: {acc:.4f}")

    # EEGNet
    eeg = build_eegnet(T, C)
    eeg.fit(X_tr, y_tr, epochs=100, batch_size=32,
            validation_split=0.1, callbacks=[es], verbose=0)
    _, acc = eeg.evaluate(X_val, y_val, verbose=0)
    results["eegnet"].append(acc)
    print(f"EEGNet       - Fold {fold+1}: {acc:.4f}")

print("\n Cross-Validation Results:")
for name, accs in results.items():
    print(f"{name}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# Visualization: CV Bar Chart
folds  = [f"Fold {i+1}" for i in range(5)]
x_pos  = np.arange(5)
width  = 0.35
plt.bar(x_pos - width/2, results["baseline"], width, label="Baseline CNN", color="steelblue")
plt.bar(x_pos + width/2, results["eegnet"],   width, label="EEGNet",       color="coral")
plt.axhline(np.mean(results["baseline"]), color="steelblue", linestyle="--", alpha=0.5)
plt.axhline(np.mean(results["eegnet"]),   color="coral",     linestyle="--", alpha=0.5)
plt.xticks(x_pos, folds)
plt.ylim(0.4, 1.0)
plt.ylabel("Accuracy")
plt.title("5-Fold Cross Validation: Baseline CNN vs EEGNet")
plt.legend(loc="upper left")
plt.savefig("results/cv_comparison.png")
plt.close()
print("Saved: cv_comparison.png")

# Final Evaluation on Held-Out Test Set
X_train, X_test, y_train, y_test = train_test_split(
    X_eeg, y, test_size=0.2, random_state=42, stratify=y)

final_model = build_eegnet(T, C)
history = final_model.fit(X_train, y_train, epochs=100, batch_size=32,
                           validation_split=0.1, callbacks=[es], verbose=1)
loss, acc = final_model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy (EEGNet): {acc:.4f}")

# Classification report (precision, recall, F1)
y_pred = (final_model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred,
      target_names=["Right Hand", "Right Foot"]))

# Visualization: Training Curve
plt.plot(history.history["accuracy"],     label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Training Curve - EEGNet")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("results/training_curve.png")
plt.close()
print("Saved: training_curve.png")

# Visualization: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Right Hand", "Right Foot"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - EEGNet")
plt.savefig("results/confusion_matrix.png")
plt.close()
print("Saved: confusion_matrix.png")

print("\n Done.")
