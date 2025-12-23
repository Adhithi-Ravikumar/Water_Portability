#  Water Potability Prediction System

This repository presents an **end-to-end, production-grade machine learning system** for **water potability assessment** using physicochemical water quality parameters. The project tackles real-world data challenges **missing values, skewed distributions, outliers and severe class imbalance** while prioritizing **model reliability, interpretability and domain relevance**.

The project  integrates engineers domain-specific features and computes the Water Quality Index (WQI); trains and compares multiple ML and ensemble models with probability calibration using Platt scaling; and leverages SHAP for model explainability. Finally, it provides actionable, data-driven recommendations for real-world water treatment and quality improvement.

 

##  Objective

The objective of this project is to design a **robust, interpretable and reliable water potability prediction system** that goes beyond simple classification. Specifically, the project aims to:

* Accurately predict whether water is **potable or non-potable** using physicochemical parameters
* Handle **real-world data challenges** such as missing values, skewness, outliers and class imbalance
* Engineer **domain-informed features** and compute the **Water Quality Index (WQI)**
* Compare multiple **machine learning and ensemble models** under different balancing strategies
* Improve **decision reliability** through probability calibration (Platt scaling)
* Provide **transparent explanations** using SHAP
* Translate predictions into **actionable, WHO-aligned water treatment recommendations**

The ultimate goal is to build a **decision-support system** for environmental monitoring, public health analysis and water treatment planning.

 

##  Key Features

###  Data Challenges Addressed

* Missing values handled using **KNN Imputation**
* Skewed feature distributions corrected via **log transformations**
* Outliers treated using **IQR clipping** and **Isolation Forest**
* Severe class imbalance addressed through:

  * Random oversampling
  * **SMOTE + Tomek Links**

 

###  Feature Engineering & Domain Logic

* Domain-driven engineered features:

  * `Water_Acidity`
  * `Mineral_Load`
  * `Organic_Risk`
  * `Disinfection_Strength`
  * `Purity_Score`
* **Water Quality Index (WQI)** computed using a standard-based weighted formulation
* WQI-based qualitative classification:

  * *Excellent → Good → Poor → Very Poor → Unsuitable*

 

###  Machine Learning Models

* Random Forest
* Extra Trees
* Gradient Boosting
* HistGradientBoosting
* AdaBoost
* XGBoost
* LightGBM
* CatBoost

Two class-balancing strategies were systematically compared:

* **Random Oversampling**
* **SMOTE–Tomek Links**

All models were tuned using **GridSearchCV** and evaluated using:

* Accuracy
* F1 Score
* ROC-AUC

 

###  Ensemble Learning

* Top-performing models combined using **Stacking Ensembles**
* Meta-learners:

  * Logistic Regression
  * Ridge Classifier

 

###  Probability Calibration

* Applied **Platt Scaling (Sigmoid Calibration)** using `CalibratedClassifierCV`
* Evaluation metrics include:

  * ROC-AUC
  * Brier Score
  * Reliability (Calibration) Curves
* Ensures **trustworthy probability estimates**, not just class predictions

 

###  Explainability with SHAP

* Fast SHAP analysis using **KernelExplainer** on the calibrated stacking model
* Global explainability:

  * Feature importance (bar plots)
  * Beeswarm plots
* Local explainability:

  * Waterfall plots for individual samples
* Chemical interpretation of features influencing potability predictions

 

###  Actionable Water Treatment Recommendations

* Threshold-based treatment recommendations derived from **WHO guidelines**
* SHAP-aware ↑ / ↓ influence indicators
* Example recommendations include:

  * pH correction strategies
  * Chloramine level adjustment
  * Turbidity reduction
  * Filtration / Reverse Osmosis (RO) guidance

 

### Custom Sample Prediction

User-facing function to:

* Input raw water chemistry values
* Compute engineered features and WQI
* Predict potability with calibrated probabilities
* Explain predictions using SHAP
* Generate actionable treatment recommendations

 

##  Results & Model Performance

###  Top 4 Individual Models (Oversampled Dataset)

| Model             | Accuracy   | F1 Score   | ROC-AUC    | Confusion Matrix       |
|      -- |    - |    - |    - |        - |
| **Extra Trees**   | **0.8575** | **0.8465** | **0.9192** | [[362, 28], [83, 306]] |
| Random Forest     | 0.8383     | 0.8376     | 0.9136     | [[328, 62], [64, 325]] |
| Gradient Boosting | 0.8139     | 0.8190     | 0.8973     | [[306, 84], [61, 328]] |
| LightGBM          | 0.8229     | 0.8258     | 0.9021     | [[314, 76], [62, 327]] |

 **Observation:**
Extra Trees achieved the strongest standalone performance, particularly in **ROC-AUC**, indicating excellent class discrimination.

 

###  Stacking Ensemble Performance

####  Base Stacking (Before Calibration)

* **Accuracy:** 0.8729
* **F1 Score:** 0.8642

The stacking ensemble outperformed all individual models in raw predictive accuracy.

 

###  Calibrated Stacking Ensemble (Platt Scaling)

| Metric      | Value  |
|    -- |    |
| Accuracy    | 0.8562 |
| F1 Score    | 0.8515 |
| ROC-AUC     | 0.8886 |
| Brier Score | 0.1219 |

 **Key Insight:**
Calibration slightly reduced accuracy but **significantly improved probability reliability**, making the model suitable for **risk-sensitive, real-world decision-making**.

 

###  Reliability & Decision Readiness

* Calibration curves show improved alignment between predicted and true probabilities
* Calibrated probabilities support:

  * Regulatory compliance
  * Public health alerts
  * Water treatment planning

 

##  Summary of Findings

* Ensemble methods outperform single models
* **Extra Trees** is the strongest standalone classifier
* **Stacking ensembles** deliver the best overall accuracy
* **Platt calibration** improves probability trustworthiness
* **SHAP + WQI** enhance interpretability and domain usability

 

##  Dataset

**Water Potability Dataset (Kaggle)**
[https://www.kaggle.com/datasets/adityakadiwal/water-potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

 

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost, LightGBM, CatBoost
* SHAP
* Matplotlib

 

## Project Goal

To move beyond simple classification and build a **reliable, interpretable and decision-ready water quality prediction system** suitable for **environmental monitoring, public health and water treatment decision support**.

 

