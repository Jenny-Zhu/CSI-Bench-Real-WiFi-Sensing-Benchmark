import h5py
import pandas as pd
import os
import json

# Examine the metadata structure
metadata_path = 'motion_source_recognition/metadata/sample_metadata.csv'
print("Examining metadata file:", metadata_path)
df = pd.read_csv(metadata_path, nrows=5)
print("Metadata columns:", df.columns.tolist())
print("Sample metadata entries:")
print(df.head())

# Examine one of the split files
split_path = 'motion_source_recognition/splits/train_id.json'
print("\nExamining split file:", split_path)
with open(split_path, 'r') as f:
    ids = json.load(f)
print(f"Number of samples in split: {len(ids)}")
print(f"Sample IDs: {ids[:5]}")

# Examine an H5 file
h5_path = 'motion_source_recognition/sub_Fan/user_4_13_2023/act_Test_1/env_4_13_2023 TestHouse/device_HP/session_708000__freq232.h5'
print("\nExamining H5 file:", h5_path)
try:
    with h5py.File(h5_path, 'r') as f:
        print("H5 keys:", list(f.keys()))
        
        # Check CSI_amps key
        if 'CSI_amps' in f:
            print("CSI_amps shape:", f['CSI_amps'].shape)
            print("CSI_amps data type:", f['CSI_amps'].dtype)
            data = f['CSI_amps'][:]
            print("CSI_amps min/max values:", data.min(), data.max())
            print("CSI_amps sample shape:", data.shape)
            
            # Show the first few values
            print("Sample data (first 5 values):", data.flatten()[:5])
        else:
            print("No 'CSI_amps' key in the H5 file")
except Exception as e:
    print(f"Error examining H5 file: {e}") 