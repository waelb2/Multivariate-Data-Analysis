# ANAD Project: Statistical Analysis of Maize Trial Data

**Authors:** Bouguessa Wail & Nemamcha Oussama
**Course:** Analyse des Données (ANAD) - Ens. N. BESSAH  
**Date:** December 2025  
**Language:** Python 3.x

---

## 📋 Project Description

This project performs a comprehensive multivariate statistical analysis on the **`agridat - butron.maize`** dataset. The goal is to analyze the relationships between genotype parents (male/female), environmental conditions, and crop yield using various statistical methods including Factorial Analysis.

### Dataset Overview

- **Source:** `agridat` package (R) - `butron.maize`
- **Observations:** 245 rows
- **Variables:**
    - **Quantitative:** `yield` (Grain yield in t/ha)
    - **Qualitative:** `gen` (Genotype), `male` (Male parent), `female` (Female parent), `env` (Environment)

---

## 🛠️ Features & Methods

The analysis pipeline is implemented in a single Python script (`main.py`) that performs the following steps:

1.  **Data Preparation:** Cleaning and formatting the dataset.
2.  **Descriptive Statistics:**
    - Univariate analysis (frequencies, means, medians).
    - Distribution plots (Histograms, Boxplots).
3.  **Bivariate Analysis:**
    - ANOVA (Analysis of Variance) for yield vs. environment.
    - Contingency tables and Chi-square tests (Male × Environment, Female × Environment).
4.  **Factorial Analysis:**
    - **MCA (Multiple Correspondence Analysis / ACM):** To analyze the structure of categorical variables (`gen`, `male`, `female`, `env`) and their relationship with yield.
    - **CA (Correspondence Analysis / AFC):** To study the specific relationship between Male and Female parent lines.

---

## 📦 Requirements

To run this project, you need Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn prince scipy scikit-learn
```

