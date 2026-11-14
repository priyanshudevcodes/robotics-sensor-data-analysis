import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVITIES = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]


def find_dataset_files():
    candidates = [
        (
            os.path.join(BASE_DIR, "train", "X_train.txt"),
            os.path.join(BASE_DIR, "train", "y_train.txt"),
            os.path.join(BASE_DIR, "test", "X_test.txt"),
            os.path.join(BASE_DIR, "test", "y_test.txt"),
        ),
        (
            os.path.join(BASE_DIR, "X_train.txt"),
            os.path.join(BASE_DIR, "y_train.txt"),
            os.path.join(BASE_DIR, "X_test.txt"),
            os.path.join(BASE_DIR, "y_test.txt"),
        ),
    ]

    for files in candidates:
        if all(os.path.exists(path) for path in files):
            return files

    raise FileNotFoundError(
        "Could not find a valid HAR dataset. Expected either "
        "<project>/train/X_train.txt and /test/X_test.txt, or root-level X_train.txt/X_test.txt. "
        "The folder currently contains corrupted/mismatched data files."
    )


def load_har_dataset():
    x_train_path, y_train_path, x_test_path, y_test_path = find_dataset_files()

    X_train = np.loadtxt(x_train_path)
    X_test = np.loadtxt(x_test_path)
    y_train = np.loadtxt(y_train_path, dtype=int).reshape(-1)
    y_test = np.loadtxt(y_test_path, dtype=int).reshape(-1)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Training/test feature matrices should be 2D arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train has {X_train.shape[1]} features, "
            f"test has {X_test.shape[1]}."
        )
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError(
            f"Training labels length mismatch: {y_train.shape[0]} labels for {X_train.shape[0]} samples."
        )
    if y_test.shape[0] != X_test.shape[0]:
        raise ValueError(
            f"Test labels length mismatch: {y_test.shape[0]} labels for {X_test.shape[0]} samples."
        )

    return X_train, y_train, X_test, y_test


def predict_knn(X_tr, y_tr, X_te, k=5):
    preds = []
    for sample in X_te:
        distances = np.linalg.norm(X_tr - sample, axis=1)
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_tr[nearest_idx]
        counts = np.bincount(nearest_labels)
        preds.append(np.argmax(counts))
    return np.array(preds)


print("Loading UCI HAR dataset...")
X_train, y_train, X_test, y_test = load_har_dataset()

# Normalize features using training statistics
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0) + 1e-8
X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std

print("Running K-Nearest Neighbors classification...")
y_pred = predict_knn(X_train_norm, y_train, X_test_norm, k=5)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Map numeric labels 1..6 to 0..5 for display and plotting
activity_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
y_test_idx = np.vectorize(activity_index.get)(y_test)
# y_pred will also be 1..6, so map it to 0..5 the same way.
y_pred_idx = np.vectorize(activity_index.get)(y_pred)

cm = pd.crosstab(
    pd.Series(y_test_idx, name="Actual"),
    pd.Series(y_pred_idx, name="Predicted"),
).reindex(index=np.arange(6), columns=np.arange(6), fill_value=0)

print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(7, 6))
plt.imshow(cm.to_numpy(), cmap="Blues")
plt.title(f"Confusion Matrix (Acc: {accuracy * 100:.1f}%)")
plt.xticks(range(6), ACTIVITIES, rotation=45)
plt.yticks(range(6), ACTIVITIES)
plt.colorbar()
plt.tight_layout()
output_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(output_path, dpi=150)
plt.close()
print(f"Saved confusion matrix to: {output_path}")
  