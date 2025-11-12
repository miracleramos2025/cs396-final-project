# CS 396 Fairness in ML - Final Project
**Group 4: Law School Admissions Fairness Analysis**

## Team Members
- Butchuong Lulhok
- Abhi Vinnakota
- Miracle Ramos

## Project Overview
This project studies fairness and accuracy trade-offs in predicting first-year law school GPA using the LSAC National Longitudinal Bar Passage Study dataset (~27,000 students with LSAT scores, undergrad GPA, race, gender, and law school outcomes). We analyze how different machine learning models perform across demographic groups and test fairness intervention techniques like reweighing to reduce bias while maintaining predictive accuracy.

## Repository Contents
- `lsac_baseline.py` - Baseline logistic regression model with fairness metrics
- `lsac_data.csv` - LSAC dataset from AIF360 library

## Getting Started
```bash
# Clone the repository
git clone https://github.com/miracleramos2025/cs396-final-project.git
cd cs396-final-project

# Install required packages
pip install pandas numpy scikit-learn fairlearn
```

## Preliminary Results
Our baseline model shows:
- **Gender**: Near-fair predictions (demographic parity diff: 0.010, equalized odds diff: 0.014)
- **Race**: Significant bias (demographic parity diff: 0.230, equalized odds diff: 0.270)
