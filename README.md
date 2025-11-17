# CS 396 Fairness in ML - Final Project
**Group 4: Law School Admissions Fairness Analysis**

## Team Members
- Butchuong Lulhok
- Abhi Vinnakota
- Miracle Ramos

## Project Overview
This project studies fairness and accuracy trade-offs in predicting first-year law school GPA using the LSAC National Longitudinal Bar Passage Study dataset (~27,000 students with LSAT scores, undergrad GPA, race, gender, and law school outcomes). We analyze how different machine learning models perform across demographic groups and test fairness intervention techniques like reweighing to reduce bias while maintaining predictive accuracy.

## Repository Contents
- **preliminary_deliverables/** - Preliminary analysis and baseline model
  - **plots/** - Visualizations (demographics, GPA distributions, outcome disparities, feature gaps)
  - `lsac_baseline.py` - Baseline logistic regression with fairness metrics
  - `lsac_eda.py` - Exploratory data analysis and quality checks
  - `lsac_visualizations.py` - Generate demographic and outcome visualizations
- `lsac_data.csv` - LSAC dataset from AIF360 library
- `README.md` - Project documentation

## Getting Started
```bash
# Clone the repository
git clone https://github.com/miracleramos2025/cs396-final-project.git
cd cs396-final-project

# Install required packages
pip install pandas numpy scikit-learn fairlearn
```

## Preliminary Results

### Exploratory Data Analysis
Our dataset contains 22,342 law students with zero missing data. We confirmed demographic encodings from AIF360 documentation: Race (0=Black, 1=White) and Gender (0=Female, 1=Male). The dataset shows severe racial imbalance with Black students representing only 7.6% of the data. EDA revealed significant feature disparities by race (LSAT gap: 0.23, UGPA gap: 0.097) but minimal disparities by gender (LSAT gap: 0.03, UGPA gap: -0.028).

### Baseline Model
A logistic regression model trained on LSAT and UGPA (excluding race and gender) achieved 62.0% accuracy and 0.647 AUC. Despite not using protected attributes directly, the model exhibits significant racial bias through proxy features:
- **Gender**: Near-fair (Demographic Parity: 0.010, Equalized Odds: 0.014)
- **Race**: Significant bias (Demographic Parity: 0.230, Equalized Odds: 0.270)

### Visualizations
Four plots document dataset composition, GPA distributions, outcome disparities, and feature gaps that explain indirect bias. Black students show consistently lower LSAT/UGPA scores and first-year GPA outcomes, explaining why models learn racial patterns even without explicit use of race as a feature.
