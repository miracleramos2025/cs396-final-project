# lsac_exploration.py
# data exploration and quality checks

import pandas as pd
import numpy as np

# load data
df = pd.read_csv("lsac_data.csv")
if "ZFYGPA" in df.columns:
    df = df.rename(columns={"ZFYGPA": "zfygpa"})

# keep only needed columns numeric
need = ["race", "gender", "lsat", "ugpa", "zfygpa"]
for c in need:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

# check 1: dataset size and missing data
print(f"\nOriginal dataset size: {len(df)} rows")
print(f"\nMissing data by column:")
print(df[need].isnull().sum())
print(f"\nRows with any missing data: {df[need].isnull().any(axis=1).sum()}")

# check complete cases
df_complete = df[need].dropna()
print(f"Complete cases (no missing values): {len(df_complete)} rows")
print(f"Data loss: {(1 - len(df_complete)/len(df)) * 100:.1f}%")

# check 2: available features?
print(f"\n" + "=" * 60)
print("AVAILABLE FEATURES")
print("=" * 60)
all_numeric = df.select_dtypes(include="number")
print(f"\nAll numeric columns in dataset: {all_numeric.columns.tolist()}")

# check 3: explore the label (zfygpa) distribution
print(f"\n" + "=" * 60)
print("LABEL DISTRIBUTION (ZFYGPA)")
print("=" * 60)
print(f"\nZFYGPA statistics:")
print(df_complete["zfygpa"].describe())

# show different threshold options
print(f"\nDifferent threshold options:")
for pct in [0.25, 0.40, 0.50, 0.60, 0.75]:
    cut = df_complete["zfygpa"].quantile(pct)
    positive_rate = (df_complete["zfygpa"] >= cut).mean()
    print(f"  Top {int((1-pct)*100):2d}% (>= {pct} quantile = {cut:.3f}): {positive_rate:.1%} labeled positive")

# check 4: demographic breakdown
print(f"\n" + "=" * 60)
print("DEMOGRAPHIC BREAKDOWN")
print("=" * 60)
print(f"\nRace distribution:")
print(df_complete["race"].value_counts().sort_index())
print(f"\nGender distribution:")
print(df_complete["gender"].value_counts().sort_index())

# check 5: feature distributions by demographics
print(f"\n" + "=" * 60)
print("FEATURE DISTRIBUTIONS BY DEMOGRAPHICS")
print("=" * 60)
print("\nLSAT by race:")
print(df_complete.groupby("race")["lsat"].describe()[["mean", "std", "min", "max"]])
print("\nUGPA by race:")
print(df_complete.groupby("race")["ugpa"].describe()[["mean", "std", "min", "max"]])

print("\nLSAT by gender:")
print(df_complete.groupby("gender")["lsat"].describe()[["mean", "std", "min", "max"]])
print("\nUGPA by gender:")
print(df_complete.groupby("gender")["ugpa"].describe()[["mean", "std", "min", "max"]])

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print("\n1. Missing Data:")
print(f"   - {(1 - len(df_complete)/len(df)) * 100:.1f}% data loss from missing values")
print("\n2. Label Balance:")
print(f"   - Using top 40% threshold yields 40% positive rate")
print("\n3. Feature Disparities (explain indirect bias):")
lsat_race_gap = df_complete.groupby("race")["lsat"].mean().diff().iloc[-1]
ugpa_race_gap = df_complete.groupby("race")["ugpa"].mean().diff().iloc[-1]
print(f"   - LSAT gap between race groups: {lsat_race_gap:.2f}")
print(f"   - UGPA gap between race groups: {ugpa_race_gap:.3f}")
print("\n   → Even without using race directly, model learns racial patterns")
print("     through LSAT and UGPA, which differ by demographic group")
