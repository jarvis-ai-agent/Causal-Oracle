import json
import numpy as np
import pandas as pd
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


def safe_json(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON serialization."""
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


def series_to_dict(s: pd.Series) -> dict:
    """Convert a pandas Series with DatetimeIndex to {date: value} dict."""
    result = {}
    for idx, val in s.items():
        key = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        result[key] = float(val) if not np.isnan(val) else None
    return result


def df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dicts with proper serialization."""
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (np.integer,)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = float(val) if not np.isnan(val) else None
            elif isinstance(val, pd.Timestamp):
                rec[col] = val.isoformat()
            else:
                rec[col] = val
        # Handle index
        idx = row.name
        if hasattr(idx, "isoformat"):
            rec["date"] = idx.isoformat()
        records.append(rec)
    return records
