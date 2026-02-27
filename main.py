import numpy as np
from load_data import build_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
data_path = "P4"
X, y = build_dataset(data_path)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Normalize the EEG data
mean = X.mean(axis=(0, 1), keepdims=True)
std = X.std(axis=(0, 1), keepdims=True) + 1e-8
X = (X - mean) / std
print("Normalization done.")

# Convert labels from {1,2} to {0,1}
if set(np.unique(y)) == {1, 2}:
    y = (y == 2).astype(np.float32)

print("Unique labels:", np.unique(y))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Baseline CNN model
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv1D(32, 5, padding="same", activation="relu"),
    Conv1D(64, 5, padding="same", activation="relu"),
    Dropout(0.3),  # reduce overfitting
    GlobalAveragePooling1D(),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model with early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Evaluate on the test set
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)