<div align="center">

# House Price Prediction

### End-to-end machine learning pipeline for the Kaggle Ames Housing competition

<br/>

<p align="center">
<img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12+-1e293b?style=for-the-badge&logo=python&logoColor=E6B17E" />
<img alt="Pandas data processing" src="https://img.shields.io/badge/Pandas-Data%20Processing-1e293b?style=for-the-badge&logo=pandas&logoColor=E6B17E" />
<img alt="NumPy numerical computing" src="https://img.shields.io/badge/NumPy-Numerical%20Computing-1e293b?style=for-the-badge&logo=numpy&logoColor=E6B17E" />
<img alt="Scikit-learn ML models" src="https://img.shields.io/badge/Scikit--Learn-ML%20Models-1e293b?style=for-the-badge&logo=scikitlearn&logoColor=E6B17E" />
</p>

<p align="center">
<img alt="XGBoost gradient boosting" src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-1e293b?style=for-the-badge&logo=xgboost&logoColor=E6B17E" />
<img alt="LightGBM boosting" src="https://img.shields.io/badge/LightGBM-Boosting-1e293b?style=for-the-badge&logo=lightgbm&logoColor=E6B17E" />
<img alt="CatBoost best model" src="https://img.shields.io/badge/CatBoost-Best%20Model-1e293b?style=for-the-badge&logo=catboost&logoColor=E6B17E" />
</p>

<p align="center">
<img alt="Kaggle competition" src="https://img.shields.io/badge/Kaggle-Competition-1e293b?style=for-the-badge&logo=kaggle&logoColor=E6B17E" />
<img alt="Leaderboard score 0.12193" src="https://img.shields.io/badge/Score-0.12193-1e293b?style=for-the-badge" />
<img alt="Task: regression" src="https://img.shields.io/badge/Task-Regression-1e293b?style=for-the-badge" />
</p>

<p align="center">
<img alt="GitHub stars" src="https://img.shields.io/github/stars/Nomusa990822/house-price-prediction?style=for-the-badge&color=1e293b" />
<img alt="GitHub forks" src="https://img.shields.io/github/forks/Nomusa990822/house-price-prediction?style=for-the-badge&color=1e293b" />
<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Nomusa990822/house-price-prediction?style=for-the-badge&color=1e293b" />
</p>

[**Overview**](#overview) · [**Pipeline**](#pipeline) · [**Methodology**](#methodology) · [**Performance**](#model-performance) · [**Blending**](#model-blending) · [**Result**](#final-submission) · [**Run It**](#how-to-run)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Model Blending](#model-blending)
- [Prediction Analysis](#prediction-analysis)
- [Final Submission](#final-submission)
- [How to Run](#how-to-run)
- [Key Insights](#key-insights)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project implements a complete machine learning pipeline for predicting residential housing prices using the **Ames Housing** dataset. The workflow covers data exploration, feature engineering, model development, evaluation, and final ensemble blending.

The objective is to minimise **Root Mean Squared Logarithmic Error (RMSLE)**, matching the Kaggle competition's evaluation metric.

<div align="center">

| Public leaderboard score | Best single-model CV RMSE | Evaluation |
|:---:|:---:|:---:|
| **0.12193** RMSLE | **0.12406** (Tuned CatBoost) | 5-fold cross-validation |

</div>

---

## Project Structure

```text
house-price-prediction/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── outputs/
│   ├── submission.csv
│   │
│   ├── modeling_outputs/
│   │   ├── model_cv_results.csv
│   │   ├── individual_model_predictions.csv
│   │   └── blend_weights.csv
│   │
│   └── catboost_info/
│       ├── learn_error.tsv
│       └── time_left.tsv
│
├── reports/
│   ├── feature_engineering.md
│   ├── model_summary.md
│   └── leaderboard_results.md
│
├── images/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Pipeline

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontFamily": "Inter, Segoe UI, Helvetica, Arial, sans-serif",
    "primaryTextColor": "#1e293b",
    "lineColor": "#64748b"
  },
  "flowchart": { "curve": "basis", "nodeSpacing": 34, "rankSpacing": 40, "htmlLabels": true }
}}%%
flowchart LR
    A[("<b>Raw Data</b><br/>Ames Housing")] --> B["<b>EDA</b><br/>01"]
    B --> C["<b>Feature<br/>Engineering</b><br/>02"]
    C --> D["<b>Baseline<br/>Models</b><br/>03"]
    D --> E["<b>5-Fold<br/>Cross Validation</b>"]
    E --> F["<b>Tuned<br/>Models</b>"]
    F --> G["<b>Model<br/>Blending</b>"]
    G --> H(["<b>Final<br/>Submission</b>"])

    classDef data  fill:#1e293b,stroke:#1e293b,color:#ffffff;
    classDef step  fill:#fdf3e7,stroke:#c98a4b,stroke-width:1.5px,color:#1e293b;
    classDef model fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#1e293b;
    classDef final fill:#c98a4b,stroke:#c98a4b,color:#ffffff;
    class A data;
    class B,C step;
    class D,E,F,G model;
    class H final;
    linkStyle default stroke:#64748b,stroke-width:1.6px
```

---

## Methodology

### 1. Data Preparation

- Missing values handled using domain-aware strategies
- Outliers analysed and treated where necessary
- Categorical variables encoded appropriately
- Numerical features transformed for stability

**Target transformation:**

$$y = \log(\text{SalePrice})$$

Predictions are converted back with:

$$\text{SalePrice} = \exp(\hat{y})$$

### 2. Feature Engineering

Key transformations include:

- Interaction features between important variables
- Polynomial expansions for non-linear relationships
- Aggregated neighbourhood-based statistics
- Log transformations for skewed distributions

The focus was on improving **signal quality** rather than increasing feature count.

### 3. Model Development

| Family | Models |
|---|---|
| **Linear** | Ridge · Lasso · ElasticNet · Bayesian Ridge |
| **Tree-based** | Random Forest · Gradient Boosting · XGBoost · LightGBM · CatBoost |

Linear models were scaled using `RobustScaler`. Tree-based models were used without scaling.

### 4. Evaluation Strategy

All models were evaluated with **5-fold cross-validation**, using:

$$\text{RMSE on } \log(\text{SalePrice})$$

This aligns directly with the Kaggle RMSLE objective.

---

## Model Performance

Cross-validation results are stored in `outputs/modeling_outputs/model_cv_results.csv`.

| Model | Baseline CV RMSE | Tuned CV RMSE | Change |
|---|:---:|:---:|:---:|
| **CatBoost** | 0.12407 | **0.12406** | -0.00001 |
| **Gradient Boosting** | 0.12729 | 0.12457 | -0.00272 |
| **Lasso** | 0.12595 | 0.12600 | +0.00005 |
| **ElasticNet** | 0.12604 | 0.12617 | +0.00013 |
| **XGBoost** | 0.12652 | 0.12656 | +0.00004 |

*Lower is better. Change is tuned minus baseline.*

**Observations**

- Boosting models capture complex non-linear relationships effectively
- Regularised linear models remain competitive thanks to strong feature engineering
- Tuning helped Gradient Boosting most; the other models were already close to their best
- Model diversity enables effective ensemble construction

---

## Model Blending

Final predictions use **inverse-RMSE weighted averaging**. Each model's weight is the inverse of its cross-validation RMSE, normalised so the weights sum to one:

$$w_i = \frac{1 / \text{RMSE}_i}{\sum_j 1 / \text{RMSE}_j}$$

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontFamily": "Inter, Segoe UI, Helvetica, Arial, sans-serif",
    "primaryTextColor": "#1e293b",
    "lineColor": "#64748b"
  },
  "flowchart": { "curve": "basis", "nodeSpacing": 30, "rankSpacing": 40, "htmlLabels": true }
}}%%
flowchart LR
    M1["<b>Linear models</b><br/>Lasso · ElasticNet"] --> W
    M2["<b>Boosting models</b><br/>CatBoost · XGBoost<br/>Gradient Boosting"] --> W
    W["<b>Inverse-RMSE<br/>weights</b><br/>w = (1/RMSE) / Σ(1/RMSE)"] --> B["<b>Weighted<br/>blend</b><br/>(log scale)"]
    B --> X["<b>exp(·)</b><br/>back to price"]
    X --> S(["<b>submission.csv</b>"])

    classDef in    fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#1e293b;
    classDef calc  fill:#fdf3e7,stroke:#c98a4b,stroke-width:1.5px,color:#1e293b;
    classDef final fill:#c98a4b,stroke:#c98a4b,color:#ffffff;
    class M1,M2 in;
    class W,B,X calc;
    class S final;
    linkStyle default stroke:#64748b,stroke-width:1.6px
```

Blend weights are stored in `outputs/modeling_outputs/blend_weights.csv`.

**Rationale**

- Stronger models contribute more to the final prediction
- Weaker models are automatically down-weighted
- Blending reduces variance and improves generalisation

---

## Prediction Analysis

Individual model predictions are available in `outputs/modeling_outputs/individual_model_predictions.csv`.

- Boosting models show tighter prediction distributions
- Linear models provide stability
- The ensemble reduces extreme prediction variance

---

## Final Submission

The submission file is `outputs/submission.csv`, in the standard Kaggle format:

| Id | SalePrice |
|:---:|:---:|
| *test house ID* | *predicted price* |

<div align="center">

### Public Leaderboard Score: **0.12193** RMSLE

</div>

This score reflects strong generalisation and effective ensemble modelling.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Nomusa990822/house-price-prediction.git

# Move into the project
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt
```

Then run the notebooks in order:

```text
01_eda  →  02_feature_engineering  →  03_modeling
```

---

## Key Insights

- Feature engineering is the primary driver of performance
- Ensemble learning outperforms individual models
- Model diversity is more valuable than model complexity
- Proper validation prevents overfitting

---

## What This Project Demonstrates

- End-to-end machine learning pipeline design
- Strong feature engineering techniques
- Robust validation strategy
- Model comparison and optimisation
- Ensemble learning and blending
- Reproducible data science workflow

---

## Future Improvements

- [ ] Advanced stacking (meta-model)
- [ ] Feature selection optimisation
- [ ] SHAP-based model explainability
- [ ] Hyperparameter optimisation using Optuna
- [ ] Deployment as a prediction API

---

## License

This project is licensed under the MIT License.

<div align="center">

<sub>[Back to top](#house-price-prediction)</sub>

</div>
