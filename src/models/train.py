from __future__ import annotations
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

from src.data.load_data import load_from_hf, load_from_csv
from src.utils.config import get_model_path, get_vectorizer_path, get_labelmap_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf", "csv"], default="hf")
    parser.add_argument("--dataset", default="banking77", help="HF dataset name (when source=hf)")
    parser.add_argument("--csv_path", default="", help="Path to CSV (when source=csv)")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()

    if args.source == "hf":
        data = load_from_hf(args.dataset)
    else:
        if not args.csv_path:
            raise ValueError("--csv_path is required when --source=csv")
        data = load_from_csv(args.csv_path, args.text_col, args.label_col)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=80_000,
        min_df=2
    )
    Xtr = vectorizer.fit_transform(data.X_train)
    Xte = vectorizer.transform(data.X_test)

    # LogisticRegression برای multiclass خوب و سریع است
    model = LogisticRegression(
        max_iter=300,
        n_jobs=None,

    )
    model.fit(Xtr, data.y_train)

    preds = model.predict(Xte)
    print(classification_report(data.y_test, preds))

    joblib.dump(model, get_model_path())
    joblib.dump(vectorizer, get_vectorizer_path())

    # برای نمایش labels در API
    labelset = sorted(list(set(data.y_train)))
    joblib.dump(labelset, get_labelmap_path())

    print(f"Saved model to: {get_model_path()}")
    print(f"Saved vectorizer to: {get_vectorizer_path()}")
    print(f"Saved labelmap to: {get_labelmap_path()}")

if __name__ == "__main__":
    main()