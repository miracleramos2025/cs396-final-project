# lsac_visualizations.py
# visualizations showing demographic breakdowns and outcome differences

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load and prepare data (from parent directory)
df = pd.read_csv("../lsac_data.csv")
if "ZFYGPA" in df.columns:
    df = df.rename(columns={"ZFYGPA": "zfygpa"})

need = ["race", "gender", "lsat", "ugpa", "zfygpa"]
for c in need:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df_complete = df[need].dropna()

# define labels for clarity
# based on AIF360 documentation: 0=unprivileged, 1=privileged
RACE_LABELS = {0: 'Black', 1: 'White'}
GENDER_LABELS = {0: 'Female', 1: 'Male'}

print("Generating visualizations...\n")


### FIGURE 1: dataset demographics
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# race distribution
race_counts = df_complete["race"].value_counts().sort_index()
ax1.bar(range(len(race_counts)), race_counts.values, color='steelblue', alpha=0.7, width=0.6)
ax1.set_xlabel("Race", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_title("Race Distribution in Dataset", fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(race_counts)))
ax1.set_xticklabels([RACE_LABELS[int(k)] for k in race_counts.index])
ax1.grid(axis='y', alpha=0.3)
# add count labels on bars
for i, (idx, val) in enumerate(race_counts.items()):
    ax1.text(i, val + 500, f'n={val:,}', ha='center', fontsize=11)
# extend y-axis to prevent label cutoff
ax1.set_ylim(0, max(race_counts.values) * 1.1)

# gender distribution
gender_counts = df_complete["gender"].value_counts().sort_index()
ax2.bar(range(len(gender_counts)), gender_counts.values, color='coral', alpha=0.7, width=0.6)
ax2.set_xlabel("Gender", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Gender Distribution in Dataset", fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(gender_counts)))
ax2.set_xticklabels([GENDER_LABELS[int(k)] for k in gender_counts.index])
ax2.grid(axis='y', alpha=0.3)
# add count labels on bars
for i, (idx, val) in enumerate(gender_counts.items()):
    ax2.text(i, val + 500, f'n={val:,}', ha='center', fontsize=11)
# extend y-axis to prevent label cutoff
ax2.set_ylim(0, max(gender_counts.values) * 1.1)

plt.tight_layout()
plt.savefig("plots/fig1_demographics.png", dpi=300, bbox_inches='tight')
print("✓ Saved: plots/fig1_demographics.png")
plt.close()


### FIGURE 2: first year GPA distribution by race (histogram)
fig2, ax = plt.subplots(figsize=(10, 6))

# plot in reverse order so Black (smaller group) appears on top
for race_val in sorted(df_complete["race"].unique(), reverse=True):
    data = df_complete[df_complete["race"] == race_val]["zfygpa"]
    ax.hist(data, alpha=0.6, label=f"{RACE_LABELS[race_val]} (n={len(data):,})", bins=40, edgecolor='black', linewidth=0.5)

ax.set_xlabel("First-Year GPA (standardized)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("First-Year GPA Distribution by Race", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("plots/fig2_gpa_distribution_by_race.png", dpi=300, bbox_inches='tight')
print("✓ Saved: plots/fig2_gpa_distribution_by_race.png")
plt.close()



### FIGURE 3: outcome disparities (ZFYGPA by demographics)
fig3, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))

# ZFYGPA by race
race_data = [df_complete[df_complete["race"] == r]["zfygpa"].values for r in sorted(df_complete["race"].unique())]
bp1 = ax1.boxplot(race_data, labels=[RACE_LABELS[int(r)] for r in sorted(df_complete["race"].unique())],
                  patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
ax1.set_xlabel("Race", fontsize=12)
ax1.set_ylabel("First-Year GPA (standardized)", fontsize=12)
ax1.set_title("First-Year GPA by Race", fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# ZFYGPA by gender
gender_data = [df_complete[df_complete["gender"] == g]["zfygpa"].values for g in sorted(df_complete["gender"].unique())]
bp2 = ax2.boxplot(gender_data, labels=[GENDER_LABELS[int(g)] for g in sorted(df_complete["gender"].unique())],
                  patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightcoral')
ax2.set_xlabel("Gender", fontsize=12)
ax2.set_ylabel("First-Year GPA (standardized)", fontsize=12)
ax2.set_title("First-Year GPA by Gender", fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle("")  # remove default title
fig3.suptitle("Outcome Disparities: First-Year GPA Varies Significantly by Race", 
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("plots/fig3_outcome_disparities.png", dpi=300, bbox_inches='tight')
print("✓ Saved: plots/fig3_outcome_disparities.png")
plt.close()


# FIGURE 4: feature disparities (source of indirect bias)
fig4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# LSAT by race
lsat_race_data = [df_complete[df_complete["race"] == r]["lsat"].values for r in sorted(df_complete["race"].unique())]
bp1 = ax1.boxplot(lsat_race_data, labels=[RACE_LABELS[int(r)] for r in sorted(df_complete["race"].unique())],
                  patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
ax1.set_xlabel("Race", fontsize=12)
ax1.set_ylabel("LSAT Score (standardized)", fontsize=12)
ax1.set_title("LSAT by Race", fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# UGPA by race
ugpa_race_data = [df_complete[df_complete["race"] == r]["ugpa"].values for r in sorted(df_complete["race"].unique())]
bp2 = ax2.boxplot(ugpa_race_data, labels=[RACE_LABELS[int(r)] for r in sorted(df_complete["race"].unique())],
                  patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightblue')
ax2.set_xlabel("Race", fontsize=12)
ax2.set_ylabel("Undergraduate GPA (standardized)", fontsize=12)
ax2.set_title("UGPA by Race", fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# LSAT by gender
lsat_gender_data = [df_complete[df_complete["gender"] == g]["lsat"].values for g in sorted(df_complete["gender"].unique())]
bp3 = ax3.boxplot(lsat_gender_data, labels=[GENDER_LABELS[int(g)] for g in sorted(df_complete["gender"].unique())],
                  patch_artist=True)
for patch in bp3['boxes']:
    patch.set_facecolor('lightcoral')
ax3.set_xlabel("Gender", fontsize=12)
ax3.set_ylabel("LSAT Score (standardized)", fontsize=12)
ax3.set_title("LSAT by Gender", fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# UGPA by gender
ugpa_gender_data = [df_complete[df_complete["gender"] == g]["ugpa"].values for g in sorted(df_complete["gender"].unique())]
bp4 = ax4.boxplot(ugpa_gender_data, labels=[GENDER_LABELS[int(g)] for g in sorted(df_complete["gender"].unique())],
                  patch_artist=True)
for patch in bp4['boxes']:
    patch.set_facecolor('lightcoral')
ax4.set_xlabel("Gender", fontsize=12)
ax4.set_ylabel("Undergraduate GPA (standardized)", fontsize=12)
ax4.set_title("UGPA by Gender", fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.suptitle("")  # remove default title
fig4.suptitle("Feature Disparities Explain Indirect Bias\n(Even without using race/gender, model learns through LSAT/UGPA)", 
              fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig("plots/fig4_feature_disparities.png", dpi=300, bbox_inches='tight')
print("✓ Saved: plots/fig4_feature_disparities.png")
plt.close()

print("\n" + "="*60)
print("All visualizations complete!")
print("="*60)
print("\nGenerated files:")
print("  1. fig1_demographics.png - Dataset composition")
print("  2. fig2_gpa_distribution_by_race.png - GPA distribution overlap")
print("  3. fig3_outcome_disparities.png - GPA by demographics")
print("  4. fig4_feature_disparities.png - LSAT/UGPA gaps (bias source)")