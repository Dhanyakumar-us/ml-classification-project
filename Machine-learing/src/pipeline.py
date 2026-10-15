"""
Customer Churn Supervised Classification Pipeline
Includes:
- Data ingestion & feature engineering
- Stratified train/test split (80/20)
- 5-Fold Stratified Cross-Validation
- Pipeline preprocessing (ColumnTransformer)
- Comparison of Logistic Regression, Random Forest, Gradient Boosting, and SVM
- Full metrics reporting: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Publication-quality visualization generation
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_and_preprocess_data(csv_path="data/churn_data.csv"):
    df = pd.read_csv(csv_path)
    
    # Feature Engineering
    df = df.copy()
    if 'phone_number' in df.columns:
        df = df.drop(columns=['phone_number'])
        
    df['total_minutes'] = (
        df['total_day_minutes'] + df['total_eve_minutes'] + 
        df['total_night_minutes'] + df['total_intl_minutes']
    )
    df['total_charge'] = (
        df['total_day_charge'] + df['total_eve_charge'] + 
        df['total_night_charge'] + df['total_intl_charge']
    )
    df['cust_service_ratio'] = df['number_customer_service_calls'] / (df['account_length'] + 1)
    df['intl_charge_ratio'] = df['total_intl_charge'] / (df['total_charge'] + 1e-6)
    
    X = df.drop(columns=['class'])
    y = df['class'].astype(int)
    
    return df, X, y


def build_preprocessor(X):
    categorical_cols = ['area_code', 'state']
    binary_cols = ['international_plan', 'voice_mail_plan']
    numeric_cols = [c for c in X.columns if c not in categorical_cols + binary_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('bin', 'passthrough', binary_cols)
        ]
    )
    return preprocessor, numeric_cols, categorical_cols, binary_cols


def plot_eda(df, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    churn_counts = df['class'].value_counts()
    churn_labels = ['Retained (0)', 'Churned (1)']
    colors = ['#2b5c8f', '#d9534f']
    
    axes[0].bar(churn_labels, churn_counts.values, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 40, f"{v:,} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold', fontsize=11)
    axes[0].set_title('Customer Churn Class Distribution', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel('Number of Customers', fontsize=11)
    axes[0].set_ylim(0, max(churn_counts.values) * 1.15)
    
    axes[1].pie(churn_counts.values, labels=churn_labels, autopct='%1.1f%%', startangle=140,
                colors=colors, explode=(0, 0.08), wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title('Target Proportion (% Imbalance)', fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_target_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Key Features by Churn
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.boxplot(x='class', y='total_day_minutes', hue='class', data=df, ax=axes[0, 0], palette=['#4575b4', '#d73027'], legend=False)
    axes[0, 0].set_title('Total Day Minutes by Churn Status', fontweight='bold')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['Retained', 'Churned'])
    axes[0, 0].set_xlabel('')
    
    sns.boxplot(x='class', y='total_charge', hue='class', data=df, ax=axes[0, 1], palette=['#4575b4', '#d73027'], legend=False)
    axes[0, 1].set_title('Total Monthly Charge by Churn Status', fontweight='bold')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['Retained', 'Churned'])
    axes[0, 1].set_xlabel('')
    
    # Customer service calls churn rate
    cs_churn = df.groupby('number_customer_service_calls')['class'].mean().reset_index()
    sns.barplot(x='number_customer_service_calls', y='class', data=cs_churn, ax=axes[1, 0], color='#e66101', edgecolor='black')
    axes[1, 0].set_title('Churn Rate by Customer Service Calls', fontweight='bold')
    axes[1, 0].set_ylabel('Churn Probability')
    axes[1, 0].set_xlabel('Number of Customer Service Calls')
    
    # International Plan churn rate
    intl_churn = df.groupby('international_plan')['class'].mean().reset_index()
    sns.barplot(x='international_plan', y='class', hue='international_plan', data=intl_churn, ax=axes[1, 1], palette=['#2b5c8f', '#d9534f'], edgecolor='black', legend=False)
    axes[1, 1].set_title('Churn Rate by International Plan', fontweight='bold')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['No Plan', 'Has Plan'])
    axes[1, 1].set_ylabel('Churn Probability')
    axes[1, 1].set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_key_feature_relationships.png"), dpi=300)
    plt.close()
    
    # 3. Correlation Heatmap
    numeric_sub = df.select_dtypes(include=[np.number]).drop(columns=['state', 'area_code', 'class'], errors='ignore')
    corr = numeric_sub.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'shrink': 0.8},
                linewidths=0.5, annot_kws={"size": 8})
    plt.title('Numeric Feature Pearson Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlation_heatmap.png"), dpi=300)
    plt.close()
    print("EDA plots saved successfully.")


def run_experiments():
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    print("Step 1: Loading data and performing feature engineering...")
    df, X, y = load_and_preprocess_data()
    plot_eda(df, output_dir=output_dir)
    
    print(f"Dataset shape: {X.shape}, Churn rate: {y.mean():.2%}")
    
    # Step 2: Stratified Train / Test Split
    print("\nStep 2: Performing Stratified 80/20 Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train samples: {len(X_train)} (Churn: {y_train.mean():.2%}) | Test samples: {len(X_test)} (Churn: {y_test.mean():.2%})")
    
    preprocessor, num_cols, cat_cols, bin_cols = build_preprocessor(X)
    
    # Step 3: Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=5,
            class_weight='balanced', random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4,
            subsample=0.85, random_state=42
        ),
        "Support Vector Machine": SVC(
            C=2.0, kernel='rbf', probability=True,
            class_weight='balanced', random_state=42
        )
    }
    
    # Step 4: Stratified 5-Fold Cross Validation (single process for stability on Windows)
    print("\nStep 3: Running 5-Fold Stratified Cross-Validation on Training Set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results_summary = {}
    
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_results_summary[name] = {
            'Accuracy (Mean ± Std)': f"{cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}",
            'Precision (Mean ± Std)': f"{cv_res['test_precision'].mean():.4f} ± {cv_res['test_precision'].std():.4f}",
            'Recall (Mean ± Std)': f"{cv_res['test_recall'].mean():.4f} ± {cv_res['test_recall'].std():.4f}",
            'F1-Score (Mean ± Std)': f"{cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}",
            'ROC-AUC (Mean ± Std)': f"{cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}",
            'raw_roc_auc_mean': cv_res['test_roc_auc'].mean(),
            'raw_f1_mean': cv_res['test_f1'].mean()
        }
        print(f"--> {name:24s} | CV ROC-AUC: {cv_res['test_roc_auc'].mean():.4f} | CV F1: {cv_res['test_f1'].mean():.4f}")
    
    cv_df = pd.DataFrame(cv_results_summary).T.drop(columns=['raw_roc_auc_mean', 'raw_f1_mean'])
    print("\nCross-Validation Comparison Table:")
    print(cv_df.to_string())
    cv_df.to_csv("reports/cross_validation_results.csv")
    
    # Step 5: Fit on Train and Evaluate on Test Set
    print("\nStep 4: Training Final Models & Evaluating on Held-Out Test Set...")
    test_metrics = []
    trained_pipelines = {}
    y_preds = {}
    y_probas = {}
    
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        
        y_preds[name] = y_pred
        y_probas[name] = y_proba
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        test_metrics.append({
            'Algorithm': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'PR-AUC': pr_auc
        })
        
    metrics_df = pd.DataFrame(test_metrics).set_index('Algorithm')
    print("\nTest Set Metrics Comparison:")
    print(metrics_df.round(4).to_string())
    metrics_df.round(4).to_csv("reports/test_metrics_comparison.csv")
    
    # Step 6: Generate Diagnostic Visualizations
    # A. ROC Curves
    plt.figure(figsize=(9, 7))
    palette = {'Logistic Regression': '#1f77b4', 'Random Forest': '#2ca02c', 'Gradient Boosting': '#d62728', 'Support Vector Machine': '#9467bd'}
    
    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, y_probas[name])
        auc = roc_auc_score(y_test, y_probas[name])
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc:.4f})", color=palette[name])
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance / Baseline (AUC = 0.5000)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10.5, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_roc_curves.png"), dpi=300)
    plt.close()
    
    # B. Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    for name in models.keys():
        prec, rec, _ = precision_recall_curve(y_test, y_probas[name])
        ap = average_precision_score(y_test, y_probas[name])
        plt.plot(rec, prec, lw=2.2, label=f"{name} (PR-AUC = {ap:.4f})", color=palette[name])
        
    baseline_ap = y_test.mean()
    plt.plot([0, 1], [baseline_ap, baseline_ap], 'k--', lw=1.2, label=f'No Skill Baseline (AP = {baseline_ap:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower left", fontsize=10.5, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_pr_curves.png"), dpi=300)
    plt.close()
    
    # C. Confusion Matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, y_pred) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    cbar=False, annot_kws={'size': 14, 'fontweight': 'bold'})
        axes[i].set_title(f"{name}\nAcc: {accuracy_score(y_test, y_pred):.3f} | F1: {f1_score(y_test, y_pred):.3f}", fontweight='bold', fontsize=11)
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        axes[i].set_ylabel('Actual Label', fontsize=10)
        axes[i].set_xticklabels(['Retained (0)', 'Churned (1)'])
        axes[i].set_yticklabels(['Retained (0)', 'Churned (1)'])
        
    plt.suptitle('Test Set Confusion Matrices Across Algorithms', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_confusion_matrices.png"), dpi=300)
    plt.close()
    
    # D. Metrics Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_plot_df = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    metrics_plot_df.plot(kind='bar', ax=ax, colormap='viridis', width=0.8, edgecolor='black', alpha=0.9)
    plt.title('Comprehensive Test Metrics Comparison Across Algorithms', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0, fontsize=10.5, fontweight='bold')
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.05), frameon=True)
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0.1:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.015),
                        ha='center', va='bottom', fontsize=8, rotation=90)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_metrics_comparison_bar.png"), dpi=300)
    plt.close()
    
    # E. Feature Importance (Random Forest and Gradient Boosting)
    rf_model = trained_pipelines['Random Forest'].named_steps['classifier']
    gb_model = trained_pipelines['Gradient Boosting'].named_steps['classifier']
    
    # Extract encoded feature names
    cat_feature_names = trained_pipelines['Random Forest'].named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names) + bin_cols
    
    rf_feat_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    gb_feat_imp = pd.Series(gb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    rf_feat_imp.sort_values().plot(kind='barh', ax=axes[0], color='#2ca02c', edgecolor='black')
    axes[0].set_title('Top 15 Features - Random Forest Gini Importance', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Relative Importance')
    
    gb_feat_imp.sort_values().plot(kind='barh', ax=axes[1], color='#d62728', edgecolor='black')
    axes[1].set_title('Top 15 Features - Gradient Boosting Feature Importance', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Relative Importance')
    
    plt.suptitle('Model Interpretability: Churn Driver Importance', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_feature_importance.png"), dpi=300)
    plt.close()
    
    print("\nAll experiments, evaluations, and figures completed successfully!")
    return metrics_df, cv_df


if __name__ == "__main__":
    run_experiments()
