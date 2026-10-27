from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "small_dataset"
CSV_FILE = BASE_DIR / "small_dataset" / "metadata.csv"
ENCODING = "ISO-8859-1"

df = pd.read_csv(CSV_FILE, encoding=ENCODING)

# 80-20 split, train-rest. Ensure speakers are distributed evenly.
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, shuffle=True, stratify=df["speaker"])

# 50-50 split, rest -> val, test. Ensure speakers are distributed evenly.
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True, stratify=temp_df["speaker"])

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding=ENCODING)
val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False, encoding=ENCODING)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding=ENCODING)

# OUTPUT FOR VISUAL FEEDBACK
print(f"Training:   {len(train_df)} recordings")
print(f"Validation: {len(val_df)} recordings")
print(f"Test:       {len(test_df)} recordings")

print("\nSpeaker distribution:")
print(pd.crosstab(df["speaker"], pd.Series("Full dataset", index=df.index)))

for name, subset in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"\n{name}:")
    print(subset["speaker"].value_counts())