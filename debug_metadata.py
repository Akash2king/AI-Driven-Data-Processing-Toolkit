#!/usr/bin/env python3
"""
Debug script to test metadata generation directly
"""

import pandas as pd
import sys
import os

# Add the backend directory to the path
sys.path.append('./backend')

try:
    from backend.app.modules.metadata_generator import metadata_generator
    
    # Load the sample data
    print("Loading sample data...")
    df = pd.read_csv("sample_survey_data.csv")
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Generate metadata
    print("\nGenerating metadata...")
    metadata = metadata_generator.generate_metadata(df)
    
    # Check the metadata structure
    print("\nMetadata structure:")
    for key, value in metadata.items():
        if isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
            if key == 'basic_info':
                print(f"    basic_info content: {value}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # Check for error
    if 'error' in metadata:
        print(f"\nERROR FOUND: {metadata['error']}")
    
    # Check specific values
    basic_info = metadata.get('basic_info', {})
    print(f"\nBasic info total_rows: {basic_info.get('total_rows', 'NOT FOUND')}")
    print(f"Basic info total_columns: {basic_info.get('total_columns', 'NOT FOUND')}")
    
    print("\nTest completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
