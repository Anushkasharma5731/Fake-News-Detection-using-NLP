"""Download the fake news dataset, train a classifier, and save its artifacts."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import joblib
import kagglehub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import sklearn
from text_utils import clean_text


DATASET = "mahdimashayekhi/fake-news-detection-dataset"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_BUNDLE_PATH", BASE_DIR / "model_bundle.joblib"))
TEXT_COLUMNS = ("text", "content", "article", "news", "title")
LABEL_COLUMNS = ("label", "class", "target", "category")


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns = {str(column).strip().lower(): column for column in df.columns}
    return next((columns[name] for name in candidates if name in columns), None)


def load_dataset(dataset_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load either a labeled CSV or the common separate Fake/True CSV files."""
    csv_files = sorted(Path(dataset_path).rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {dataset_path}")

    frames = [(file, pd.read_csv(file)) for file in csv_files]
    labeled = []
    for file, frame in frames:
        label_column = _find_column(frame, LABEL_COLUMNS)
        text_column = _find_column(frame, TEXT_COLUMNS)
        if label_column is not None and text_column is not None:
            labeled.append(frame.rename(columns={label_column: "label", text_column: "text"})[["text", "label"]])

    if labeled:
        data = pd.concat(labeled, ignore_index=True)
    else:
        fake_files = [(file, frame) for file, frame in frames if "fake" in file.name.lower()]
        true_files = [(file, frame) for file, frame in frames if "true" in file.name.lower()]
        if not fake_files or not true_files:
            raise ValueError(
                "Could not find labeled data or both Fake and True CSV files. "
                f"CSV files found: {[file.name for file, _ in frames]}"
            )

        parts = []
        for group, label in ((fake_files, "fake"), (true_files, "real")):
            for file, frame in group:
                text_column = _find_column(frame, TEXT_COLUMNS)
                if text_column is None:
                    raise ValueError(f"No article text column found in {file.name}")
                part = frame[[text_column]].rename(columns={text_column: "text"})
                part["label"] = label
                parts.append(part)
        data = pd.concat(parts, ignore_index=True)

    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["text"].str.strip().ne("")]
    if data.empty:
        raise ValueError("The dataset contains no usable article text and labels")
    if data["label"].nunique() < 2:
        raise ValueError("Training requires at least two distinct label classes")
    return data


def main() -> None:
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download(DATASET)
    df = load_dataset(dataset_path)
    print(f"Loaded {len(df):,} articles; labels: {df['label'].value_counts().to_dict()}")

    print("Cleaning text data...")
    df["cleaned_text"] = df["text"].map(clean_text)
    df = df[df["cleaned_text"].str.len().gt(0)]
    if df.empty:
        raise ValueError("No usable text remains after cleaning the dataset")
    if df["label"].value_counts().min() < 2:
        raise ValueError("Each label class needs at least two rows for a train/test split")

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42,
        stratify=df["label"],
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_df=0.7, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")
    model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print(f"\nModel accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_name(f".{MODEL_PATH.name}.tmp")
    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "labels": [str(label) for label in model.classes_],
        "dataset": DATASET,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
    }
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        os.replace(temporary_path, MODEL_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Model bundle saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
