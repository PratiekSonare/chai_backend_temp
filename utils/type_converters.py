"""
Utility functions for converting NumPy/Pandas types to native Python types
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy and Pandas types to native Python types
    for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, pandas type, etc.)
    
    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return convert_numpy_types(obj.to_dict())
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    else:
        return obj


def safe_int(value: Any) -> int:
    """Safely convert value to int, handling numpy types"""
    if pd.isna(value):
        return 0
    if isinstance(value, (np.integer, np.floating)):
        return int(value)
    return int(value)


def safe_float(value: Any) -> float:
    """Safely convert value to float, handling numpy types"""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return float(value)
