from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    artifacts: Path = root / "artifacts"

def get_artifacts_dir() -> Path:
    p = os.getenv("ARTIFACTS_DIR", str(Paths().artifacts))
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_model_path() -> Path:
    return get_artifacts_dir() / "model.joblib"

def get_vectorizer_path() -> Path:
    return get_artifacts_dir() / "vectorizer.joblib"

def get_labelmap_path() -> Path:
    return get_artifacts_dir() / "labelmap.joblib"