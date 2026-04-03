import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("PIPELINE_DATA_DIR", str(BASE_DIR / "data")))
ARTIFACTS_DIR = Path(os.getenv("PIPELINE_ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
DB_PATH = Path(os.getenv("PIPELINE_DB_PATH", str(BASE_DIR / "pipeline.db")))
TIMESFM_MODEL_ID = os.getenv("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
TIMESFM_DEVICE = os.getenv("TIMESFM_DEVICE", "cpu")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "SPY"],
    "target": "AAPL_ret",
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "max_lag": 5,
    "alpha": 0.05,
    "indep_test": "rcot",
    "n_regimes": 3,
    "horizon": 5,
    "context_length": 1024,
    "initial_capital": 100000.0,
    "causal_retrain_interval": 60,
    "forecast_retrain_interval": 5,
}
