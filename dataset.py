
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = "NLP project/Dataset/ARC-Easy-Train.csv"
output_file = BASE_DIR / "data" / "processed" / "arc_easy_pilot_20.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().dropna(how="all")

if len(df) < 20:
    raise ValueError("Dataset contains fewer than 20 valid rows.")

selected_df = df.sample(
    n=20,
    random_state=42
).reset_index(drop=True)

output_file.parent.mkdir(parents=True, exist_ok=True)

selected_df.to_csv(output_file, index=False)

print("\nSelected questions:", len(selected_df))
print("\nSelected dataset:")
print(selected_df.to_string(index=False))

print(f"\nDataset saved to: {output_file}")
