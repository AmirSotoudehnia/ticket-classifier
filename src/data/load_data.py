from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv
from pathlib import Path

@dataclass
class DatasetBundle:
    X_train: List[str]
    y_train: List[str]
    X_test: List[str]
    y_test: List[str]

def load_from_csv(csv_path: str, text_col: str = "text", label_col: str = "label") -> DatasetBundle:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if text_col not in r or label_col not in r:
                raise ValueError(f"CSV must contain columns: {text_col}, {label_col}")
            rows.append((r[text_col].strip(), r[label_col].strip()))

    # split ساده 90/10
    n = len(rows)
    if n < 50:
        raise ValueError("CSV dataset too small. Provide at least ~50 rows.")
    split = int(n * 0.9)
    train = rows[:split]
    test = rows[split:]

    X_train = [t for t, _ in train]
    y_train = [y for _, y in train]
    X_test = [t for t, _ in test]
    y_test = [y for _, y in test]
    return DatasetBundle(X_train, y_train, X_test, y_test)

def load_from_hf(dataset_name: str = "banking77") -> DatasetBundle:
    """
    گزینه سریع: دیتاست 'banking77' (intent classification) شبیه تیکت‌های پشتیبانی است.
    نیاز به اینترنت در زمان اجرا دارد.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets package not available. Install requirements.txt") from e

    ds = load_dataset(dataset_name)
    # banking77: columns -> 'text', 'label' (label int)
    # label names:
    label_names = ds["train"].features["label"].names

    def convert(split):
        X = [x["text"] for x in ds[split]]
        y = [label_names[x["label"]] for x in ds[split]]
        return X, y

    X_train, y_train = convert("train")
    X_test, y_test = convert("test")
    return DatasetBundle(X_train, y_train, X_test, y_test)