# Supervised Classification Model Evaluation Report: Customer Churn Prediction

**Project:** Supervised Machine Learning Benchmark & Churn Classification  
**Date:** October 2026  
**Artifacts Generated:**  
- Jupyter Notebook: [`customer_churn_classification.ipynb`](customer_churn_classification.ipynb)
- Pipeline Script: [`src/pipeline.py`](src/pipeline.py)
- Benchmark Data: [`data/churn_data.csv`](data/churn_data.csv)
- Generated Figures: [`figures/`](figures/)

---

## 1. Executive Summary

Customer attrition (churn) directly impacts company profitability and revenue growth. In the telecommunications sector, acquiring a new customer costs 5 to 7 times more than retaining an existing one. 

This project developed, validated, and evaluated four supervised classification algorithms on a customer dataset comprising **5,000 subscriber accounts** with an empirical churn rate of **14.14%**:
1. **Logistic Regression** (L2 Regularized Baseline with Cost-Sensitive Weighting)
2. **Random Forest Classifier** (Bagging Ensemble of 150 Decision Trees)
3. **Gradient Boosting Classifier** (Sequential Gradient Boosted Decision Trees)
4. **Support Vector Machine (SVC)** (Non-Linear Radial Basis Function Kernel)

### Primary Verdict:
**Random Forest Classifier** is selected as the top-performing production model. On the unseen 1,000-customer test partition, it attained:
- **Accuracy:** `97.60%`
- **Precision:** `98.35%` (minimizes costly false-alarm retention incentives)
- **Recall:** `84.40%` (captures the vast majority of at-risk subscribers)
- **F1-Score:** `90.84%`
- **ROC-AUC:** `0.9267`
- **PR-AUC:** `0.8997` (substantially exceeding the 0.141 baseline)

---

## 2. Dataset & Exploratory Data Analysis (EDA)

The dataset contains customer demographic data, plan configurations, usage metrics broken down by time period (day, evening, night, international), and customer service interactions.

### Class Imbalance Profile
- **Total Records:** 5,000
- **Retained Customers (`class = 0`):** 4,293 (85.86%)
- **Churned Customers (`class = 1`):** 707 (14.14%)
- **Imbalance Ratio:** ~ 6.1 : 1

*Key Insight:* Because 85.9% of customers do not churn, a naive dummy model predicting "Retained" for every customer would achieve 85.9% accuracy but 0.0% recall, failing completely at the business objective. Therefore, **Recall, Precision, F1-Score, ROC-AUC, and PR-AUC** were designated as primary evaluation criteria over raw accuracy.

### Key EDA Findings
1. **Customer Service Interaction Spike:** Customers making 4 or more calls to customer service exhibit a churn rate over **45% to 60%** (compared to only ~11% for customers with 1-2 calls).
2. **International Plan Discrepancy:** Subscribers with an active international plan churn at more than **3.5x** the rate of non-international plan subscribers (~42% vs ~11%), highlighting dissatisfaction with international call rates or roaming plans.
3. **Daytime Call Volume & Charges:** Heavy daytime callers accumulate higher monthly charges, creating high sensitivity to competing carrier promotions.

---

## 3. Data Preprocessing & Leak-Free Pipeline

To prevent data snooping and target leakage, preprocessing and transformations were strictly encapsulated within `scikit-learn` `Pipeline` and `ColumnTransformer` constructs.

### Feature Engineering
1. **`total_minutes`**: Aggregate sum of daytime, evening, night, and international call minutes.
2. **`total_charge`**: Total monthly billed amount across all periods.
3. **`cust_service_ratio`**: Customer service call count divided by account tenure.
4. **`intl_charge_ratio`**: Proportion of the monthly bill spent on international calls.
5. **Identifier Removal**: Arbitrary identifier `phone_number` dropped.

### Transformation Pipeline
- **Numeric Features (17 continuous variables):** Standardized using `StandardScaler` ($\mu = 0, \sigma = 1$).
- **Categorical Features (`area_code`, `state`):** Encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- **Binary Flags (`international_plan`, `voice_mail_plan`):** Direct binary pass-through.

### Validation Scheme
- **Stratified 80/20 Train/Test Split:**
  - Training set: 4,000 customers (566 churners, 14.15%)
  - Held-out test set: 1,000 customers (141 churners, 14.10%)
- **5-Fold Stratified Cross-Validation:** Out-of-fold validation on the training partition to estimate generalization error and variance across splits.

---

## 4. Cross-Validation Performance (Training Set)

Across 5 stratified folds on the 4,000-sample training partition:

| Algorithm | Accuracy (Mean ± SD) | Precision (Mean ± SD) | Recall (Mean ± SD) | F1-Score (Mean ± SD) | ROC-AUC (Mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9760 ± 0.0050** | 0.9662 ± 0.0099 | **0.8605 ± 0.0315** | **0.9100 ± 0.0198** | **0.9242 ± 0.0236** |
| **Random Forest** | 0.9727 ± 0.0050 | **0.9852 ± 0.0160** | 0.8199 ± 0.0294 | 0.8947 ± 0.0202 | 0.9192 ± 0.0156 |
| **Support Vector Machine (SVC)** | 0.9163 ± 0.0095 | 0.6890 ± 0.0363 | 0.7474 ± 0.0345 | 0.7165 ± 0.0300 | 0.9072 ± 0.0151 |
| **Logistic Regression** | 0.7920 ± 0.0110 | 0.3829 ± 0.0149 | 0.7667 ± 0.0653 | 0.5100 ± 0.0227 | 0.8380 ± 0.0243 |

---

## 5. Final Evaluation on Held-Out Test Set (1,000 Samples)

The final pipelines were trained on the entire training set and scored on the held-out test set:

| Algorithm | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC | PR-AUC (Average Precision) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9760** | **0.9835** | 0.8440 | **0.9084** | **0.9267** | **0.8997** |
| **Gradient Boosting** | 0.9740 | 0.9528 | **0.8582** | 0.9030 | 0.9255 | 0.8982 |
| **Support Vector Machine (SVC)** | 0.9080 | 0.6485 | 0.7589 | 0.6993 | 0.9064 | 0.7890 |
| **Logistic Regression** | 0.7860 | 0.3728 | 0.7589 | 0.5000 | 0.8346 | 0.4916 |

---

## 6. Confusion Matrix & Error Trade-Off Analysis

On 1,000 test customers (859 Retained, 141 Churned):

```
Logistic Regression:
  [TN=679, FP=180]    False Positives: 180 (Low Precision: 37.3%)
  [FN= 34, TP=107]    False Negatives: 34  (Recall: 75.9%)

Random Forest:
  [TN=857, FP=  2]    False Positives: 2   (High Precision: 98.4%)
  [FN= 22, TP=119]    False Negatives: 22  (Recall: 84.4%)

Gradient Boosting:
  [TN=853, FP=  6]    False Positives: 6   (High Precision: 95.3%)
  [FN= 20, TP=121]    False Negatives: 20  (Recall: 85.8%)

Support Vector Machine (SVC):
  [TN=801, FP= 58]    False Positives: 58  (Precision: 64.9%)
  [FN= 34, TP=107]    False Negatives: 34  (Recall: 75.9%)
```

### Business Impact of the Trade-Off:
1. **False Positives (FP):** Customers wrongly predicted to churn. Providing retention offers (e.g. $50 bill credits or free device upgrades) to non-churning customers leads to unnecessary budget expenditure.
   - Logistic Regression generated **180 false alarms**, which would waste significant promotional capital.
   - Random Forest generated only **2 false alarms** out of 859 retained subscribers!
2. **False Negatives (FN):** Customers who were predicted to stay but churned. 
   - Both Random Forest (22 FN) and Gradient Boosting (20 FN) successfully identified ~85% of all churning accounts.

---

## 7. Model Interpretability & Churn Drivers

Inspecting tree Gini importance and feature contributions identified the top predictive signals:
1. **`total_charge` & `total_day_minutes` (~30% importance):** Accounts with heavy daytime talk minutes pay high per-minute charges and are significantly more price-sensitive.
2. **`number_customer_service_calls` & `cust_service_ratio` (~24% importance):** Customers with unresolved network, device, or billing problems call repeatedly; 4 or more calls indicates a critical churn tipping point.
3. **`international_plan` (~18% importance):** High churn among international plan subscribers signals uncompetitive overseas call pricing or roaming quality.
4. **`total_eve_minutes` & `voice_mail_plan`:** Usage engagement indicators.

---

## 8. Strategic Recommendations for Business Deployment

1. **Production Deployment with Random Forest:**
   - Deploy `RandomForestClassifier` with an automated weekly scoring batch pipeline for all active accounts.
   - Flag accounts with predicted probability $P(\text{churn}) \ge 0.50$ for the customer success priority queue.
2. **Proactive Intervention at Call 3:**
   - Establish an automated CRM workflow rule: when a subscriber opens their 3rd customer service ticket within 60 days, trigger a senior customer service specialist follow-up to resolve underlying pain points before churn intent solidifies.
3. **International Plan Tariff Review:**
   - Conduct product and pricing review on international call packages to match competitor rates and reduce the 42% attrition rate in this segment.

---

## 9. Verification & How to Reproduce

All code, data, tests, and plots are packaged in the repository:

```bash
# 1. Run the complete pipeline script
py src/pipeline.py

# 2. View generated diagnostic plots
# Located in the 'figures/' directory:
# - figures/01_target_distribution.png
# - figures/02_key_feature_relationships.png
# - figures/03_correlation_heatmap.png
# - figures/04_roc_curves.png
# - figures/05_pr_curves.png
# - figures/06_confusion_matrices.png
# - figures/07_metrics_comparison_bar.png
# - figures/08_feature_importance.png

# 3. Open or launch Jupyter Notebook
py -m jupyter notebook customer_churn_classification.ipynb
```
