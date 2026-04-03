import json
import joblib
import pickle
from pathlib import Path
from typing import Any, Optional
from config import ARTIFACTS_DIR
from utils.logging import get_logger
from utils.serialization import NumpyEncoder

logger = get_logger(__name__)


def run_dir(run_id: str) -> Path:
    d = ARTIFACTS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_artifact(run_id: str, name: str, obj: Any):
    """Save artifact. JSON for serializable objects, pickle fallback."""
    d = run_dir(run_id)
    try:
        path = d / f"{name}.json"
        with open(path, "w") as f:
            json.dump(obj, f, cls=NumpyEncoder)
        logger.info(f"Saved artifact {name} for run {run_id}")
    except Exception as e:
        path = d / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved artifact {name} (pickle) for run {run_id}")


def load_artifact(run_id: str, name: str) -> Optional[Any]:
    d = run_dir(run_id)
    json_path = d / f"{name}.json"
    pkl_path = d / f"{name}.pkl"
    if json_path.exists():
        try:
            with open(json_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load artifact {name} as JSON: {e}")
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load artifact {name} as pickle: {e}")
    return None


def save_joblib(run_id: str, name: str, obj: Any):
    path = run_dir(run_id) / f"{name}.joblib"
    joblib.dump(obj, path)


def load_joblib(run_id: str, name: str) -> Optional[Any]:
    path = run_dir(run_id) / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def artifact_exists(run_id: str, name: str) -> bool:
    d = run_dir(run_id)
    return (d / f"{name}.json").exists() or (d / f"{name}.pkl").exists()
