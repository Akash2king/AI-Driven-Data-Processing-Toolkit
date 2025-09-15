import pandas as pd
import numpy as np
import json

# Test the conversion function
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'item'):  # Handle other numpy scalars
        return obj.item()
    else:
        try:
            # Try to convert to native Python type
            if hasattr(obj, 'dtype'):
                return obj.item() if hasattr(obj, 'item') else str(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

# Create test data
df = pd.read_csv("d:/statathon/test_survey_data.csv")

# Generate simple metadata
metadata = {
    "total_rows": int(len(df)),
    "total_columns": int(len(df.columns)),
    "column_names": list(df.columns),
    "missing_values": {str(col): int(df[col].isnull().sum()) for col in df.columns},
    "column_types": {str(col): str(df[col].dtype) for col in df.columns}
}

# Convert and test JSON serialization
converted = convert_numpy_types(metadata)
print("Original metadata keys:", list(metadata.keys()))
print("Converted metadata keys:", list(converted.keys()))

try:
    json_str = json.dumps(converted)
    print("JSON serialization successful!")
    print("Sample output:", json_str[:200] + "...")
except Exception as e:
    print("JSON serialization failed:", e)
