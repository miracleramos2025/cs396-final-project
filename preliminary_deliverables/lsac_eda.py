# lsac_exploration.py
# Data exploration and quality checks

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("lsac_data.csv")
if "ZFYGPA" in df.columns:
    df = df.rename(columns={"ZFYGPA": "zfygpa"})

# Keep only needed columns numeric
need = ["race", "gender", "lsat", "ugpa", "zfygpa"]
for c in need:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

# Check 1: Dataset size and missing data
print(f"\nOriginal dataset size: {len(df)} rows")
print(f"\nMissing data by column:")
print(df[need].isnull().sum())
print(f"\nRows with any missing data: {df[need].isnull().any(axis=1).sum()}")

# Check complete cases
df_complete = df[need].dropna()
print(f"Complete cases (no missing values): {len(df_complete)} rows")
print(f"Data loss: {(1 - len(df_complete)/len(df)) * 100:.1f}%")

# Check 2: What features are available?
print(f"\n" + "=" * 60)
print("AVAILABLE FEATURES")
print("=" * 60)
all_numeric = df.select_dtypes(include="number")
print(f"\nAll numeric columns in dataset: {all_numeric.columns.tolist()}")

# Check 3: Examine the label (zfygpa) distribution
print(f"\n" + "=" * 60)
print("LABEL DISTRIBUTION (ZFYGPA)")
print("=" * 60)
print(f"\nZFYGPA statistics:")
print(df_complete["zfygpa"].describe())

# Show different threshold options
print(f"\nDifferent threshold options:")
for pct in [0.25, 0.40, 0.50, 0.60, 0.75]:
    cut = df_complete["zfygpa"].quantile(pct)
    positive_rate = (df_complete["zfygpa"] >= cut).mean()
    print(f"  Top {int((1-pct)*100):2d}% (>= {pct} quantile = {cut:.3f}): {positive_rate:.1%} labeled positive")

# Check 4: Demographic breakdown
print(f"\n" + "=" * 60)
print("DEMOGRAPHIC BREAKDOWN")
print("=" * 60)

# Verify encodings
print("\nVerifying demographic encodings:")
print(f"Unique race values: {sorted(df_complete['race'].unique())}")
print(f"Unique gender values: {sorted(df_complete['gender'].unique())}")

print(f"\nRace distribution:")
race_counts = df_complete["race"].value_counts().sort_index()
print(race_counts)
print(f"\nGender distribution:")
gender_counts = df_complete["gender"].value_counts().sort_index()
print(gender_counts)

# Based on AIF360 documentation and distribution patterns
print("\n" + "-" * 60)
print("DEMOGRAPHIC ENCODING (from AIF360 documentation):")
print("-" * 60)
print("Race: 0 = Black (unprivileged), 1 = White (privileged)")
print("Gender: 0 = Female (unprivileged), 1 = Male (privileged)")
print("\nVerification:")
print(f"  Race 0 (Black): {race_counts[0.0]:,} students (minority group)")
print(f"  Race 1 (White): {race_counts[1.0]:,} students (majority group)")
print(f"  Gender 0 (Female): {gender_counts[0.0]:,} students")
print(f"  Gender 1 (Male): {gender_counts[1.0]:,} students")

# Check 5: Feature distributions by demographics
print(f"\n" + "=" * 60)
print("FEATURE DISTRIBUTIONS BY DEMOGRAPHICS")
print("=" * 60)
print("\nLSAT by race:")
lsat_by_race = df_complete.groupby("race")["lsat"].describe()[["mean", "std", "min", "max"]]
print(lsat_by_race)
print("\nUGPA by race:")
ugpa_by_race = df_complete.groupby("race")["ugpa"].describe()[["mean", "std", "min", "max"]]
print(ugpa_by_race)

print("\nLSAT by gender:")
lsat_by_gender = df_complete.groupby("gender")["lsat"].describe()[["mean", "std", "min", "max"]]
print(lsat_by_gender)
print("\nUGPA by gender:")
ugpa_by_gender = df_complete.groupby("gender")["ugpa"].describe()[["mean", "std", "min", "max"]]
print(ugpa_by_gender)

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
print(f"   - LSAT gap between Black and White students: {lsat_race_gap:.2f}")
print(f"   - UGPA gap between Black and White students: {ugpa_race_gap:.3f}")
print("\n   → Even without using race directly, model learns racial patterns")
print("     through LSAT and UGPA, which differ by demographic group")
print("\n4. Gender Disparities:")
lsat_gender_gap = df_complete.groupby("gender")["lsat"].mean().diff().iloc[-1]
ugpa_gender_gap = df_complete.groupby("gender")["ugpa"].mean().diff().iloc[-1]
print(f"   - LSAT gap between Female and Male students: {lsat_gender_gap:.2f}")
print(f"   - UGPA gap between Female and Male students: {ugpa_gender_gap:.3f}")
print("   → Much smaller gaps explain why baseline model shows fairness for gender")
