# Supervised Machine Learning: Customer Churn Classification

A complete, production-grade supervised machine learning pipeline and interactive Jupyter notebook for predicting customer churn.

## 🚀 Overview

- **Dataset**: Customer Churn (5,000 subscriber accounts, 21 initial attributes).
- **Target**: `class` (0 = Retained [85.86%], 1 = Churned [14.14%]).
- **Algorithms Compared**:
  1. **Logistic Regression** (L2 Regularized Baseline, Balanced Class Weights)
  2. **Random Forest Classifier** (Bagging Ensemble of 150 Decision Trees)
  3. **Gradient Boosting Classifier** (Sequential Gradient Boosted Trees)
  4. **Support Vector Machine (SVC)** (RBF Non-linear Kernel)
- **Validation**: Stratified 80/20 train/test split + 5-Fold Stratified Cross-Validation on the training partition.

---

## 📊 Summary of Results

### 5-Fold Stratified Cross-Validation (Training Partition, N=4,000)

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9760 ± 0.005** | 0.9662 ± 0.010 | **0.8605 ± 0.032** | **0.9100 ± 0.020** | **0.9242 ± 0.024** |
| **Random Forest** | 0.9727 ± 0.005 | **0.9852 ± 0.016** | 0.8199 ± 0.029 | 0.8947 ± 0.020 | 0.9192 ± 0.016 |
| **Support Vector Machine** | 0.9163 ± 0.010 | 0.6890 ± 0.036 | 0.7474 ± 0.035 | 0.7165 ± 0.030 | 0.9072 ± 0.015 |
| **Logistic Regression** | 0.7920 ± 0.011 | 0.3829 ± 0.015 | 0.7667 ± 0.065 | 0.5100 ± 0.023 | 0.8380 ± 0.024 |

### Final Evaluation on Held-Out Test Set (N=1,000)

| Algorithm | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9760** | **0.9835** | 0.8440 | **0.9084** | **0.9267** | **0.8997** |
| **Gradient Boosting** | 0.9740 | 0.9528 | **0.8582** | 0.9030 | 0.9255 | 0.8982 |
| **Support Vector Machine** | 0.9080 | 0.6485 | 0.7589 | 0.6993 | 0.9064 | 0.7890 |
| **Logistic Regression** | 0.7860 | 0.3728 | 0.7589 | 0.5000 | 0.8346 | 0.4916 |

---

## 📁 Repository Structure

```
├── data/
│   └── churn_data.csv                    # Complete 5,000-row churn dataset
├── figures/
│   ├── 01_target_distribution.png        # Class distribution and imbalance analysis
│   ├── 02_key_feature_relationships.png  # Boxplots and churn rates by feature
│   ├── 03_correlation_heatmap.png        # Feature correlation matrix
│   ├── 04_roc_curves.png                 # Multi-model ROC comparison curves
│   ├── 05_pr_curves.png                  # Precision-Recall curves (PR-AUC)
│   ├── 06_confusion_matrices.png         # Test set 2x2 confusion matrices
│   ├── 07_metrics_comparison_bar.png     # Metric comparison bar chart
│   └── 08_feature_importance.png         # Top churn drivers (Gini importance)
├── reports/
│   ├── cross_validation_results.csv      # Out-of-fold CV metrics
│   └── test_metrics_comparison.csv       # Test set metrics
├── src/
│   └── pipeline.py                       # Modular ML pipeline
├── customer_churn_classification.ipynb   # Fully executed interactive notebook
├── MODEL_EVALUATION_REPORT.md            # Detailed model selection and business report
└── README.md                             # Project overview and instructions
```

---

## ⚙️ How to Run

### 1. Run the Complete Pipeline
```bash
py src/pipeline.py
```

### 2. Launch the Jupyter Notebook
```bash
py -m jupyter notebook customer_churn_classification.ipynb
```
