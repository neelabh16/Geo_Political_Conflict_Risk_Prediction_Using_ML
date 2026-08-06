# 🌍 Geo-Political Conflict Risk Prediction

> **An end-to-end, leakage-free machine learning pipeline for predicting geopolitical conflict risk using political, economic, military, social, and humanitarian indicators.**

A comprehensive framework that combines **supervised learning**, **regression**, **clustering**, **explainable AI**, and **temporal forecasting** to estimate the probability and severity of geopolitical conflicts across countries.

---

# 📖 Overview

Geopolitical conflicts emerge from a complex interaction of political instability, economic stress, social fragmentation, military posture, environmental pressures, and historical tensions.

This project aims to answer one fundamental question:

> **Can we predict the risk of future conflicts using measurable structural indicators?**

The notebook implements a complete machine-learning pipeline that:

* Predicts conflict-risk categories
* Estimates conflict intensity
* Discovers geopolitical archetypes
* Explains model decisions using SHAP
* Prevents temporal and feature leakage
* Provides an inference system for future predictions

The entire pipeline is designed to mimic real-world forecasting conditions.

---

# 🎯 Research Objectives

The project investigates the following questions:

* Which factors are the strongest predictors of war?
* Can political instability predict future conflicts?
* How important are economic crises?
* Do military indicators improve prediction accuracy?
* Can machine learning identify countries at risk before conflict begins?
* What patterns emerge across different geopolitical regions?

---

# 🧠 Problem Formulation

The task is formulated as a **multi-class classification problem**.

The model predicts one of four conflict-risk categories.

| Class | Meaning       |
| ----- | ------------- |
| 0     | Low Risk      |
| 1     | Moderate Risk |
| 2     | High Risk     |
| 3     | Extreme Risk  |

Target variable:

```text
conflict_risk_label
```

---

# 📊 Dataset

The project uses a large geopolitical dataset containing structural, behavioral, economic, and military indicators collected over multiple countries and years.

## Dataset dimensions

The dataset includes information such as:

* Country
* Region
* Year
* Political indicators
* Economic indicators
* Military indicators
* Humanitarian indicators
* Environmental indicators
* Historical conflict metrics

---

# 📂 Feature Categories

The features are divided into four major groups.

---

## 🔴 Structural Causes (Pre-conflict Features)

These variables represent the root causes of war.

### Political indicators

* Government stability index
* Democracy score
* Corruption index
* Political rights score
* Civil liberties score
* Coup attempts
* Political instability score

### Economic indicators

* GDP per capita
* Inflation rate
* Unemployment rate
* Poverty rate
* External debt
* Economic stress index

### Social indicators

* Ethnic fractionalization
* Religious tensions
* Refugee pressure
* Human development index
* Education index

### Environmental indicators

* Water stress
* Climate vulnerability
* Resource scarcity

---

## 🟡 Escalation Signals

These variables capture rising tensions before conflict.

* Border disputes
* Military spending
* Arms imports
* Regional tensions
* Foreign interventions
* Diplomatic breakdown indicators

---

## 🔵 Concurrent Conflict Signals

These features occur during conflict and are quarantined to avoid leakage.

Examples:

* Conflict intensity score
* Casualty counts
* Active insurgencies
* Refugee displacement

---

## 🟢 Consequence Features

These variables represent the aftermath of conflict.

Examples:

* Humanitarian crises
* Economic collapse
* Infrastructure destruction
* Migration shocks

---

# 🔥 Leakage Firewall

One of the core contributions of this project is the **Leakage Firewall**.

The dataset is divided into two separate tracks.

---

## Track A — Conflict Risk Prediction

Used for forecasting future conflict.

Includes:

* Structural causes
* Political indicators
* Economic indicators
* Military posture
* Social tensions

Excludes:

* Conflict outcomes
* Consequence variables
* Future information

---

## Track B — Conflict Intensity Analysis

Used for understanding ongoing conflicts.

Includes:

* Conflict intensity
* Humanitarian consequences
* War severity indicators

---

# ⚙️ Data Cleaning Pipeline

The preprocessing stage includes:

✅ Numeric conversion

✅ Currency and percentage cleaning

✅ Sentinel-value removal

✅ Missing-value handling

✅ Outlier clipping

✅ Regime normalization

✅ Feature validation

✅ Leakage removal

---

# 🧹 Data Cleaning Steps

| Step | Description                |
| ---- | -------------------------- |
| 1    | Numeric coercion           |
| 2    | Remove sentinel values     |
| 3    | Missing-value handling     |
| 4    | Winsorization              |
| 5    | Type correction            |
| 6    | Regime normalization       |
| 7    | Duplicate removal          |
| 8    | Feature consistency checks |
| 9    | Final validation           |

---

# 🔧 Feature Engineering

Several composite indicators are engineered to improve predictive performance.

## Political indicators

```text
political_instability_score
```

Combines:

* government instability
* corruption
* democracy erosion
* civil liberties

---

## Economic indicators

```text
economic_stress_index
```

Combines:

* inflation
* unemployment
* poverty
* debt burden

---

## Humanitarian indicators

```text
humanitarian_stress_index
```

Combines:

* refugee pressure
* food insecurity
* migration pressure

---

## Military indicators

```text
military_posture_score
```

Combines:

* military spending
* arms imports
* force mobilization

---

## Fragility score

```text
fragility_composite
```

Measures overall state fragility.

---

# 📊 Exploratory Data Analysis

The notebook generates:

* Class distributions
* Missing-value heatmaps
* Correlation matrices
* Risk distributions
* Regional comparisons
* Political-instability trends
* Economic-stress distributions
* Military posture analysis

---

# ⏳ Temporal Train-Test Split

Traditional random splitting causes future leakage.

The project uses a temporal split:

```text
70% → Training

15% → Validation

15% → Test
```

Old observations are used to predict future observations.

This simulates real-world geopolitical forecasting.

---

# ⚖️ Scaling and Balancing

The pipeline uses:

* StandardScaler
* Class weighting
* Balanced sampling
* Distribution checks

to address class imbalance.

---

# 🤖 Models Trained

The notebook compares multiple machine-learning algorithms.

## Classification models

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Support Vector Machine
* Gradient Boosting
* Extra Trees

---

# 📈 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Macro
* ROC-AUC
* Balanced accuracy
* Confusion matrices

The primary metric is:

```text
F1 Macro
```

because geopolitical datasets are often imbalanced.

---

# 🔬 Hyperparameter Optimization

The best-performing model is optimized using:

```text
Optuna Bayesian Optimization
```

The tuning process searches for:

* learning rate
* tree depth
* regularization
* estimators
* subsampling ratio

---

# 🏆 Stacking Ensemble

The project builds a meta-model using stacking.

Architecture:

```text
Base Models

↓

Random Forest

↓

XGBoost

↓

LightGBM

↓

Meta Learner

↓

Final Prediction
```

The ensemble improves:

* robustness
* stability
* generalization

---

# 📊 Model Evaluation

The notebook generates:

* Confusion matrices
* ROC curves
* Precision-recall curves
* Fold comparisons
* Calibration plots
* Error distributions

---

# 🧠 Explainability with SHAP

The project uses SHAP to explain predictions.

SHAP answers questions such as:

* Why is a country classified as high risk?
* Which indicators matter most?
* Which features reduce risk?
* What drives extreme-risk predictions?

Generated plots:

* SHAP summary plots
* Beeswarm plots
* Feature importance charts
* Dependence plots

---

# 🔵 Regression Module

In addition to classification, the notebook predicts:

```text
conflict_intensity_score
```

using regression models.

Models include:

* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

Evaluation metrics:

* RMSE
* MAE
* R² Score

---

# 🔍 Unsupervised Learning

The notebook discovers geopolitical archetypes using clustering.

---

## K-Means Clustering

Countries are grouped into clusters based on:

* political instability
* economic stress
* military posture
* fragility
* humanitarian pressure

---

## PCA Visualization

Principal Component Analysis reduces high-dimensional indicators into a two-dimensional geopolitical map.

Possible clusters:

* Stable democracies
* Fragile states
* Resource-dependent economies
* Active conflict zones

---

# 🎯 Inference System

The notebook exposes a prediction function:

```python
predict_risk(...)
```

that estimates:

* conflict-risk category
* probability distribution
* exponential risk score

Example:

```python
predict_risk(
    democracy_score=0.45,
    corruption_index=0.82,
    military_spending=4.7,
    unemployment_rate=18.0
)
```

Output:

```text
Predicted Risk: High

Probability: 0.84
```

---

# 💾 Pipeline Serialization

The trained pipeline is exported using:

```python
pickle
```

Saved components:

* scaler
* model
* feature list
* label encoder
* preprocessing metadata

---

# 📁 Project Structure

```text
Geo_Political_Conflict_Risk_Prediction/

│
├── Geo_Political_Conflict_Risk_Prediction.ipynb
├── geopolitical_conflict_coloured.xlsx
├── README.md
│
├── plots/
│   ├── class_distribution.png
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   ├── roc_curve.png
│   └── pca_clusters.png
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── metadata.pkl
│
└── results/
    ├── metrics.csv
    ├── regression_results.csv
    └── feature_importance.csv
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/geopolitical-conflict-risk-prediction.git

cd geopolitical-conflict-risk-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
optuna
shap
scipy
```

---

# ▶️ Running the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
Geo_Political_Conflict_Risk_Prediction.ipynb
```

Run all cells sequentially.

---

# 🔄 Pipeline Workflow

```text
Load Data

↓

Data Cleaning

↓

Leakage Firewall

↓

Feature Engineering

↓

EDA

↓

Temporal Split

↓

Scaling

↓

Model Training

↓

Hyperparameter Tuning

↓

Stacking Ensemble

↓

Evaluation

↓

SHAP Explainability

↓

Regression

↓

Clustering

↓

Inference

↓

Serialization
```

---

# ⚠️ Limitations

* Geopolitical systems are highly nonlinear.
* Correlation does not imply causation.
* External shocks are difficult to model.
* Data quality varies across countries.
* Rare conflicts create class imbalance.

---

# 🚀 Future Work

Potential extensions:

* Graph neural networks
* Transformer-based forecasting
* Satellite imagery integration
* News sentiment analysis
* Causal inference
* Reinforcement learning
* Real-time conflict monitoring
* Time-series transformers

---
---

# 👨‍💻 Author

**Neelabh**

Machine Learning • Data Science • Computational Geopolitics • Explainable AI

---

# ⭐ Support

If you found this project useful, consider starring the repository and contributing to future research in geopolitical forecasting.
