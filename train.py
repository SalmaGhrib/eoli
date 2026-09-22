import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
# Test du déclenchement GitHub Actions
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# Test du réentraînement automatique GitHub Actions
CLASS_TO_TARGET = {
    "BENIGN_OR_NORMAL": 0,
    "MALIGNANT": 1,
}

def find_dataset():
    candidates = []
    for key in ("DATASET_PATH", "DATA_PATH"):
        v = os.getenv(key)
        if v:
            candidates.append(Path(v))

    candidates += [
        Path("/workspace/breast_ultrasound_dataset_128.zip"),
        Path("/workspace/data/breast_ultrasound_dataset_128.zip"),
        Path("/workspace/datasets/breast_ultrasound_dataset_128.zip"),
        Path("breast_ultrasound_dataset_128.zip"),
    ]

    for p in candidates:
        if p.exists():
            return p

    for root in (Path("/workspace"), Path("/app"), Path(".")):
        if root.exists():
            for p in root.rglob("*.zip"):
                if p.is_file():
                    return p
    raise FileNotFoundError("No dataset ZIP found.")

def extract_if_needed(dataset_path):
    if dataset_path.is_dir():
        return dataset_path, None

    tmp = Path(tempfile.mkdtemp(prefix="breastmnist_"))
    with zipfile.ZipFile(dataset_path, "r") as z:
        z.extractall(tmp)
    return tmp, tmp

def image_to_features(path):
    img = Image.open(path).convert("L").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    feat = hog(
        arr,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return feat

def load_split(root, split):
    X, y = [], []
    split_dir = root / split

    for class_name, target in CLASS_TO_TARGET.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                X.append(image_to_features(img_path))
                y.append(target)

    if not X:
        raise RuntimeError(f"No images found for split {split}")

    return np.asarray(X), np.asarray(y)

def main():
    dataset_path = find_dataset()
    print(f"dataset_path={dataset_path}")

    root, tmp = extract_if_needed(dataset_path)

    try:
        X_train, y_train = load_split(root, "train")
        X_val, y_val = load_split(root, "val")
        X_test, y_test = load_split(root, "test")

        print(f"train_samples={len(y_train)}")
        print(f"val_samples={len(y_val)}")
        print(f"test_samples={len(y_test)}")

        # Medium-complexity image classifier:
        # HOG image descriptors + RBF SVM.
        model = Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        model.fit(X_fit, y_fit)

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, proba)

        # Platform-readable metrics
        print(f"accuracy={accuracy:.6f}")
        print(f"precision={precision:.6f}")
        print(f"recall={recall:.6f}")
        print(f"f1_score={f1:.6f}")
        print(f"roc_auc={auc:.6f}")

        out = Path("outputs")
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.pkl")

        meta = {
            "task": "binary-image-classification",
            "modality": "breast ultrasound",
            "classes": {"0": "BENIGN_OR_NORMAL", "1": "MALIGNANT"},
            "feature_extractor": "HOG",
            "classifier": "RBF SVM",
            "image_size": [128, 128],
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": auc,
            },
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("model_saved=outputs/model.pkl")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
