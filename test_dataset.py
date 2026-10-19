# test_dataset.py
import pandas as pd
import numpy as np
from utils import preprocess_pixels

csv_path = "data/fer2013.csv"  # adjust path if needed

try:
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}, Columns: {df.columns.tolist()}")
    print(f"First row: {df.iloc[0].to_dict()}")
    
    # Test preprocessing on first pixel string
    pixel_str = df["pixels"].iloc[0]
    arr = preprocess_pixels(pixel_str)
    print(f"Preprocessed shape: {arr.shape}")
except Exception as e:
    print(f"Error: {e}")