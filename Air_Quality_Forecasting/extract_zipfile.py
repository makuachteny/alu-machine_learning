""" Extract the zip file containing the time series data. """

import os
import zipfile
import shutil

file_path = "../time_series/assignment-1-time-series-forecasting-may-2025.zip"
extract_path = "../time_series/"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    
    print("Extracted files:")
    for file in zip_ref.namelist():
        print(file)