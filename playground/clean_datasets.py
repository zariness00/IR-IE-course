import os
import glob
import pandas as pd
from dotenv import load_dotenv
import re

load_dotenv("playground/genius_api.env")
csv_folder = os.getenv("CSV_FOLDER")


keywords = [
    "Remix",
    "remix",
    "Deluxe",
    "deluxe",
    "Live",
    "live",
    "Acoustic",
    "acoustic",
    "Demo",
    "Demo"
    "version",
    "Version",
    "Acapella",
    "acapella",
    "Instrumental",
    "instrumental",
    "edit",
    "Edit",
    "Radio Edit",
    "radio edit",
    "Radio edit",
    "Mix",
    "mix",
    "cover",
    "Cover"
    ]
escaped = [re.escape(k) for k in keywords]
pattern = rf"({'|'.join(escaped)})"

output_folder = os.path.join(csv_folder, "Cleaned_csvs")
os.makedirs(output_folder, exist_ok=True)


for csv_path in glob.glob(os.path.join(csv_folder, "*.csv")):
    df = pd.read_csv(csv_path)
    mask = df["Title"].str.contains(pattern, regex=True, na=False)
    cleaned_df = df[~mask].copy()
    out_path = os.path.join(output_folder, os.path.basename(csv_path))
    cleaned_df.to_csv(out_path, index=False)
    print("Checkpoint!!!")
    print(f"Processed {os.path.basename(csv_path)} → {os.path.basename(out_path)}")