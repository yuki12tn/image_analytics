import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, log_loss, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def find_dataset_path(dataset_name, search_paths):
    for path in search_paths:
        if os.path.exists(path):
            print(f"Found '{dataset_name}' dataset at: {path}")
            return path
    raise FileNotFoundError(f"Dataset '{dataset_name}' not found in the provided paths.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'train': [os.path.join(BASE_DIR, "train_2_split.csv")],
    'validation': [os.path.join(BASE_DIR, "val_2_split.csv")],
    'test_v2': [os.path.join(BASE_DIR, "test_2_split.csv")]
}

try:
    train_path = find_dataset_path('train', DATASETS['train'])
    val_path = find_dataset_path('validation', DATASETS['validation'])
    test_v2_path = find_dataset_path('test_v2', DATASETS['test_v2'])
except FileNotFoundError as e:
    print(e)
    exit(1)

# Load datasets
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_v2_df = pd.read_csv(test_v2_path)

# Prepare features and labels
X_train_full = train_df.drop(columns=['Unnamed: 0', 'path', 'label'])
y_train_full = train_df['label']
X_val_full = val_df.drop(columns=['Unnamed: 0', 'path', 'label'])
y_val_full = val_df['label']
X_test_v2_full = test_v2_df.drop(columns=['Unnamed: 0', 'path', 'label'])
y_test_v2_full = test_v2_df['label']

# Sampling
SAMPLE_SIZE = 10000
if len(X_train_full) > SAMPLE_SIZE:
    X_train_sample = X_train_full.sample(n=SAMPLE_SIZE, random_state=42)
    y_train_sample = y_train_full.loc[X_train_sample.index]
else:
    X_train_sample = X_train_full
    y_train_sample = y_train_full

if len(X_val_full) > SAMPLE_SIZE:
    X_val_sample = X_val_full.sample(n=SAMPLE_SIZE, random_state=42)
    y_val_sample = y_val_full.loc[X_val_sample.index]
else:
    X_val_sample = X_val_full
    y_val_sample = y_val_full

if len(X_test_v2_full) > SAMPLE_SIZE:
    X_test_v2_sample = X_test_v2_full.sample(n=SAMPLE_SIZE, random_state=42)
    y_test_v2_sample = y_test_v2_full.loc[X_test_v2_sample.index]
else:
    X_test_v2_sample = X_test_v2_full
    y_test_v2_sample = y_test_v2_full

# Best parameters found from previous grid search
best_params = {
    'C': 0.01,
    'penalty': 'l2',
    'solver': 'lbfgs'
}
best_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=best_params['solver'],
        max_iter=500,
        n_jobs=-1
    )
)

# Train the model
best_model.fit(X_train_sample, y_train_sample)

# Predictions on Validation Set (Test Set 1)
y_val_pred = best_model.predict(X_val_sample)
y_val_proba = best_model.predict_proba(X_val_sample)

# Compute evaluation metrics for Test Set 1
accuracy = accuracy_score(y_val_sample, y_val_pred)
balanced_acc = balanced_accuracy_score(y_val_sample, y_val_pred)
precision = precision_score(y_val_sample, y_val_pred, average='weighted', zero_division=0)
recall = recall_score(y_val_sample, y_val_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val_sample, y_val_pred, average='weighted', zero_division=0)
mcc = matthews_corrcoef(y_val_sample, y_val_pred)
cohen_kappa = cohen_kappa_score(y_val_sample, y_val_pred)
logloss = log_loss(y_val_sample, y_val_proba)

# Generate classification report for Test Set 1
class_report_full = classification_report(y_val_sample, y_val_pred)

print("Model has been run successfully.\n")
print("Validation Accuracy (Test Set 1):", f"{accuracy:.4f}")
print("Balanced Accuracy (Test Set 1):", f"{balanced_acc:.4f}")
print("Precision (Weighted) (Test Set 1):", f"{precision:.4f}")
print("Recall (Weighted) (Test Set 1):", f"{recall:.4f}")
print("F1 Score (Weighted) (Test Set 1):", f"{f1:.4f}")
print("Matthews Correlation Coefficient (Test Set 1):", f"{mcc:.4f}")
print("Cohen's Kappa (Test Set 1):", f"{cohen_kappa:.4f}")
print("Log Loss (Test Set 1):", f"{logloss:.4f}\n")

print("Classification Report for Test Set 1:\n")
print(class_report_full)

# Save classification report to a file (optional)
# with open('classification_report_test_set_1.txt', 'w') as f:
#     f.write(class_report_full)

# Analyze Top Misclassified Classes for Test Set 1
report_dict_full = classification_report(y_val_sample, y_val_pred, output_dict=True)
report_df_full = pd.DataFrame(report_dict_full).transpose().iloc[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
sorted_report_full = report_df_full.sort_values(by='f1-score')
N = 10
top_misclassified_full = sorted_report_full.head(N)

selected_classes_full = top_misclassified_full.index.astype(int)

mask_full = y_val_sample.isin(selected_classes_full)
y_val_selected_full = y_val_sample[mask_full]
y_val_pred_selected_full = y_val_pred[mask_full]

conf_matrix_full = confusion_matrix(y_val_selected_full, y_val_pred_selected_full, labels=selected_classes_full)

# Plot Confusion Matrix for Top Misclassified Classes - Test Set 1
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_full, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes_full, yticklabels=selected_classes_full)
plt.title('Confusion Matrix for Top Misclassified Classes - Logistic Regression (Test Set 1)')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.tight_layout()
plt.show()

# Plot Bar Plot of F1-Scores for Top Misclassified Classes - Test Set 1
plt.figure(figsize=(12, 6))
sns.barplot(x=top_misclassified_full.index, y='f1-score', data=top_misclassified_full, palette='viridis')
plt.title(f'Top {N} Misclassified Classes (Lowest F1-scores) - Logistic Regression (Test Set 1)')
plt.xlabel('Class')
plt.ylabel('F1-score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature Importance Analysis for Logistic Regression
coefficients_full = best_model.named_steps['logisticregression'].coef_

# Compute average absolute coefficients for feature importance
feature_importance_full = np.mean(np.abs(coefficients_full), axis=0)

feature_names_full = X_train_sample.columns
feature_importance_df_full = pd.DataFrame({
    'Feature': feature_names_full,
    'Importance': feature_importance_full
})

top_features_full = feature_importance_df_full.sort_values(by='Importance', ascending=False).head(N)

# Plot Top Important Features - Test Set 1
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_full, palette='viridis')
plt.title(f'Top {N} Important Features - Logistic Regression (Test Set 1)')
plt.xlabel('Average Absolute Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ROC Curves for Selected Classes - Test Set 1
classes_full = np.unique(y_val_sample)
y_val_binary_full = label_binarize(y_val_sample, classes=classes_full)
n_classes_full = y_val_binary_full.shape[1]

selected_classes = selected_classes_full.tolist()[:3]  # Select top 3 misclassified classes for example

plt.figure(figsize=(10, 7))
for cls in selected_classes:
    cls_index = np.where(classes_full == cls)[0][0]
    
    fpr, tpr, _ = roc_curve(y_val_binary_full[:, cls_index], y_val_proba[:, cls_index])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves for Selected Classes - Logistic Regression (Test Set 1)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
for cls in selected_classes:
    cls_index = np.where(classes_full == cls)[0][0]
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_val_binary_full[:, cls_index], y_val_proba[:, cls_index])
    avg_precision = average_precision_score(y_val_binary_full[:, cls_index], y_val_proba[:, cls_index])
    
    plt.plot(recall_curve, precision_curve, label=f'Class {cls} (AP = {avg_precision:.2f})')

plt.title('Precision-Recall Curves for Selected Classes - Logistic Regression (Test Set 1)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

train_sizes_full, train_scores_full, val_scores_full = learning_curve(
    estimator=best_model,
    X=X_train_sample,
    y=y_train_sample,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

train_scores_mean_full = np.mean(train_scores_full, axis=1)
val_scores_mean_full = np.mean(val_scores_full, axis=1)
train_scores_std_full = np.std(train_scores_full, axis=1)
val_scores_std_full = np.std(val_scores_full, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_full, train_scores_mean_full, 'o-', label='Training Score', color='r')
plt.plot(train_sizes_full, val_scores_mean_full, 'o-', label='Cross-Validation Score', color='g')
plt.fill_between(train_sizes_full, train_scores_mean_full - train_scores_std_full,
                 train_scores_mean_full + train_scores_std_full, alpha=0.1, color='r')
plt.fill_between(train_sizes_full, val_scores_mean_full - val_scores_std_full,
                 val_scores_mean_full + val_scores_std_full, alpha=0.1, color='g')
plt.title('Learning Curve - Logistic Regression Classifier (Test Set 1)')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()

misclassified_indices_full = y_val_sample != y_val_pred

num_misclassified_full = misclassified_indices_full.sum()
print(f"Number of misclassified samples in Test Set 1: {num_misclassified_full} out of {len(y_val_sample)}")

misclassification_rate_full = num_misclassified_full / len(y_val_sample)
print(f"Misclassification Rate in Test Set 1: {misclassification_rate_full:.2%}")

metrics_df_full = pd.DataFrame({
    'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', "Cohen's Kappa", 'Log Loss'],
    'Value': [accuracy, balanced_acc, precision, recall, f1, mcc, cohen_kappa, logloss]
})

print("\nSummary of Evaluation Metrics for Test Set 1:")
print(metrics_df_full)

X_test_v2 = X_test_v2_sample
y_test_v2 = y_test_v2_sample

y_test_v2_pred = best_model.predict(X_test_v2)
y_test_v2_proba = best_model.predict_proba(X_test_v2)

accuracy_test_v2 = accuracy_score(y_test_v2, y_test_v2_pred)
balanced_acc_test_v2 = balanced_accuracy_score(y_test_v2, y_test_v2_pred)
precision_test_v2 = precision_score(y_test_v2, y_test_v2_pred, average='weighted', zero_division=0)
recall_test_v2 = recall_score(y_test_v2, y_test_v2_pred, average='weighted', zero_division=0)
f1_test_v2 = f1_score(y_test_v2, y_test_v2_pred, average='weighted', zero_division=0)
mcc_test_v2 = matthews_corrcoef(y_test_v2, y_test_v2_pred)
cohen_kappa_test_v2 = cohen_kappa_score(y_test_v2, y_test_v2_pred)
logloss_test_v2 = log_loss(y_test_v2, y_test_v2_proba)

class_report_test_v2 = classification_report(y_test_v2, y_test_v2_pred)

print("Evaluation on Test Set 2:\n")
print("Validation Accuracy (Test Set 2):", f"{accuracy_test_v2:.4f}")
print("Balanced Accuracy (Test Set 2):", f"{balanced_acc_test_v2:.4f}")
print("Precision (Weighted) (Test Set 2):", f"{precision_test_v2:.4f}")
print("Recall (Weighted) (Test Set 2):", f"{recall_test_v2:.4f}")
print("F1 Score (Weighted) (Test Set 2):", f"{f1_test_v2:.4f}")
print("Matthews Correlation Coefficient (Test Set 2):", f"{mcc_test_v2:.4f}")
print("Cohen's Kappa (Test Set 2):", f"{cohen_kappa_test_v2:.4f}")
print("Log Loss (Test Set 2):", f"{logloss_test_v2:.4f}\n")

print("Classification Report for Test Set 2:\n")
print(class_report_test_v2)
eport_dict_full = classification_report(y_val_sample, y_val_pred, output_dict=True)
report_df_full = pd.DataFrame(report_dict_full).transpose().iloc[:-3]  
sorted_report_full = report_df_full.sort_values(by='f1-score')
N = 10
top_misclassified_full = sorted_report_full.head(N)

selected_classes_full = top_misclassified_full.index.astype(int)

mask_full = y_val_sample.isin(selected_classes_full)
y_val_selected_full = y_val_sample[mask_full]
y_val_pred_selected_full = y_val_pred[mask_full]

conf_matrix_full = confusion_matrix(y_val_selected_full, y_val_pred_selected_full, labels=selected_classes_full)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_full, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes_full, yticklabels=selected_classes_full)
plt.title('Confusion Matrix for Top Misclassified Classes - Logistic Regression (Test Set 1)')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=top_misclassified_full.index, y='f1-score', data=top_misclassified_full, palette='viridis')
plt.title(f'Top {N} Misclassified Classes (Lowest F1-scores) - Logistic Regression (Test Set 1)')
plt.xlabel('Class')
plt.ylabel('F1-score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

coefficients_full = best_model.named_steps['logisticregression'].coef_

feature_importance_full = np.mean(np.abs(coefficients_full), axis=0)

feature_names_full = X_train_sample.columns
feature_importance_df_full = pd.DataFrame({
    'Feature': feature_names_full,
    'Importance': feature_importance_full
})

top_features_full = feature_importance_df_full.sort_values(by='Importance', ascending=False).head(N)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_full, palette='viridis')
plt.title(f'Top {N} Important Features - Logistic Regression (Test Set 1)')
plt.xlabel('Average Absolute Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

classes_full = np.unique(y_val_sample)
y_val_binary_full = label_binarize(y_val_sample, classes=classes_full)
n_classes_full = y_val_binary_full.shape[1]

selected_classes = selected_classes_full.tolist()[:3] 

plt.figure(figsize=(10, 7))
for cls in selected_classes:
    cls_index = np.where(classes_full == cls)[0][0]
    
    fpr, tpr, _ = roc_curve(y_val_binary_full[:, cls_index], y_val_proba[:, cls_index])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves for Selected Classes - Logistic Regression (Test Set 1)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
for cls in selected_classes:
    cls_index = np.where(classes_full == cls)[0][0]
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_val_binary_full[:, cls_index], y_val_proba[:, cls_index])
    avg_precision = average_precision_score(y_val_binary_full[:, cls_index], y_val_proba[:, cls_index])
    
    plt.plot(recall_curve, precision_curve, label=f'Class {cls} (AP = {avg_precision:.2f})')

plt.title('Precision-Recall Curves for Selected Classes - Logistic Regression (Test Set 1)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

train_sizes_full, train_scores_full, val_scores_full = learning_curve(
    estimator=best_model,
    X=X_train_sample,
    y=y_train_sample,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

train_scores_mean_full = np.mean(train_scores_full, axis=1)
val_scores_mean_full = np.mean(val_scores_full, axis=1)
train_scores_std_full = np.std(train_scores_full, axis=1)
val_scores_std_full = np.std(val_scores_full, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_full, train_scores_mean_full, 'o-', label='Training Score', color='r')
plt.plot(train_sizes_full, val_scores_mean_full, 'o-', label='Cross-Validation Score', color='g')
plt.fill_between(train_sizes_full, train_scores_mean_full - train_scores_std_full,
                 train_scores_mean_full + train_scores_std_full, alpha=0.1, color='r')
plt.fill_between(train_sizes_full, val_scores_mean_full - val_scores_std_full,
                 val_scores_mean_full + val_scores_std_full, alpha=0.1, color='g')
plt.title('Learning Curve - Logistic Regression Classifier (Test Set 1)')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()

misclassified_indices_full = y_val_sample != y_val_pred

num_misclassified_full = misclassified_indices_full.sum()
print(f"Number of misclassified samples in Test Set 1: {num_misclassified_full} out of {len(y_val_sample)}")

misclassification_rate_full = num_misclassified_full / len(y_val_sample)
print(f"Misclassification Rate in Test Set 1: {misclassification_rate_full:.2%}")

metrics_df_full = pd.DataFrame({
    'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', "Cohen's Kappa", 'Log Loss'],
    'Value': [accuracy, balanced_acc, precision, recall, f1, mcc, cohen_kappa, logloss]
})

print("\nSummary of Evaluation Metrics for Test Set 1:")
print(metrics_df_full)

X_test_v2 = X_test_v2_sample
y_test_v2 = y_test_v2_sample

y_test_v2_pred = best_model.predict(X_test_v2)
y_test_v2_proba = best_model.predict_proba(X_test_v2)

accuracy_test_v2 = accuracy_score(y_test_v2, y_test_v2_pred)
balanced_acc_test_v2 = balanced_accuracy_score(y_test_v2, y_test_v2_pred)
precision_test_v2 = precision_score(y_test_v2, y_test_v2_pred, average='weighted', zero_division=0)
recall_test_v2 = recall_score(y_test_v2, y_test_v2_pred, average='weighted', zero_division=0)
f1_test_v2 = f1_score(y_test_v2, y_test_v2_pred, average='weighted', zero_division=0)
mcc_test_v2 = matthews_corrcoef(y_test_v2, y_test_v2_pred)
cohen_kappa_test_v2 = cohen_kappa_score(y_test_v2, y_test_v2_pred)
logloss_test_v2 = log_loss(y_test_v2, y_test_v2_proba)

class_report_test_v2 = classification_report(y_test_v2, y_test_v2_pred)

print("Evaluation on Test Set 2:\n")
print("Validation Accuracy (Test Set 2):", f"{accuracy_test_v2:.4f}")
print("Balanced Accuracy (Test Set 2):", f"{balanced_acc_test_v2:.4f}")
print("Precision (Weighted) (Test Set 2):", f"{precision_test_v2:.4f}")
print("Recall (Weighted) (Test Set 2):", f"{recall_test_v2:.4f}")
print("F1 Score (Weighted) (Test Set 2):", f"{f1_test_v2:.4f}")
print("Matthews Correlation Coefficient (Test Set 2):", f"{mcc_test_v2:.4f}")
print("Cohen's Kappa (Test Set 2):", f"{cohen_kappa_test_v2:.4f}")
print("Log Loss (Test Set 2):", f"{logloss_test_v2:.4f}\n")

print("Classification Report for Test Set 2:\n")
print(class_report_test_v2)

report_dict_test_v2 = classification_report(y_test_v2, y_test_v2_pred, output_dict=True)
report_df_test_v2 = pd.DataFrame(report_dict_test_v2).transpose().iloc[:-3]  
sorted_report_test_v2 = report_df_test_v2.sort_values(by='f1-score')
top_misclassified_test_v2 = sorted_report_test_v2.head(N)

selected_classes_test_v2 = top_misclassified_test_v2.index.astype(int)

mask_test_v2 = y_test_v2.isin(selected_classes_test_v2)
y_test_v2_selected = y_test_v2[mask_test_v2]
y_test_v2_pred_selected = y_test_v2_pred[mask_test_v2]

conf_matrix_test_v2 = confusion_matrix(y_test_v2_selected, y_test_v2_pred_selected, labels=selected_classes_test_v2)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_test_v2, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes_test_v2, yticklabels=selected_classes_test_v2)
plt.title('Confusion Matrix for Top Misclassified Classes - Logistic Regression (Test Set 2)')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=top_misclassified_test_v2.index, y='f1-score', data=top_misclassified_test_v2, palette='viridis')
plt.title(f'Top {N} Misclassified Classes (Lowest F1-scores) - Logistic Regression (Test Set 2)')
plt.xlabel('Class')
plt.ylabel('F1-score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

classes_test_v2 = np.unique(y_test_v2)
y_test_v2_binary = label_binarize(y_test_v2, classes=classes_test_v2)
n_classes_test_v2 = y_test_v2_binary.shape[1]

selected_classes_test_v2_list = selected_classes_test_v2.tolist()[:3]  

plt.figure(figsize=(10, 7))
for cls in selected_classes_test_v2_list:
    cls_index = np.where(classes_test_v2 == cls)[0][0]
    
    fpr, tpr, _ = roc_curve(y_test_v2_binary[:, cls_index], y_test_v2_proba[:, cls_index])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves for Selected Classes - Logistic Regression (Test Set 2)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
for cls in selected_classes_test_v2_list:
    cls_index = np.where(classes_test_v2 == cls)[0][0]
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_v2_binary[:, cls_index], y_test_v2_proba[:, cls_index])
    avg_precision = average_precision_score(y_test_v2_binary[:, cls_index], y_test_v2_proba[:, cls_index])
    
    plt.plot(recall_curve, precision_curve, label=f'Class {cls} (AP = {avg_precision:.2f})')

plt.title('Precision-Recall Curves for Selected Classes - Logistic Regression (Test Set 2)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

train_sizes_test_v2, train_scores_test_v2, val_scores_test_v2 = learning_curve(
    estimator=best_model,
    X=X_train_sample,
    y=y_train_sample,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

train_scores_mean_test_v2 = np.mean(train_scores_test_v2, axis=1)
val_scores_mean_test_v2 = np.mean(val_scores_test_v2, axis=1)
train_scores_std_test_v2 = np.std(train_scores_test_v2, axis=1)
val_scores_std_test_v2 = np.std(val_scores_test_v2, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_test_v2, train_scores_mean_test_v2, 'o-', label='Training Score', color='r')
plt.plot(train_sizes_test_v2, val_scores_mean_test_v2, 'o-', label='Cross-Validation Score', color='g')
plt.fill_between(train_sizes_test_v2, train_scores_mean_test_v2 - train_scores_std_test_v2,
                 train_scores_mean_test_v2 + train_scores_std_test_v2, alpha=0.1, color='r')
plt.fill_between(train_sizes_test_v2, val_scores_mean_test_v2 - val_scores_std_test_v2,
                 val_scores_mean_test_v2 + val_scores_std_test_v2, alpha=0.1, color='g')
plt.title('Learning Curve - Logistic Regression Classifier (Test Set 2)')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()

misclassified_indices_test_v2 = y_test_v2 != y_test_v2_pred

num_misclassified_test_v2 = misclassified_indices_test_v2.sum()
print(f"Number of misclassified samples in Test Set 2: {num_misclassified_test_v2} out of {len(y_test_v2)}")

misclassification_rate_test_v2 = num_misclassified_test_v2 / len(y_test_v2)
print(f"Misclassification Rate in Test Set 2: {misclassification_rate_test_v2:.2%}")

metrics_df_test_v2 = pd.DataFrame({
    'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', "Cohen's Kappa", 'Log Loss'],
    'Value': [accuracy_test_v2, balanced_acc_test_v2, precision_test_v2, recall_test_v2, f1_test_v2, mcc_test_v2, cohen_kappa_test_v2, logloss_test_v2]
})

print("\nSummary of Evaluation Metrics for Test Set 2:")
print(metrics_df_test_v2)

accuracy_gap = accuracy - accuracy_test_v2
f1_gap = f1 - f1_test_v2

print("\nPerformance Gap Between Test Set 1 and Test Set 2:")
print(f"Accuracy Drop: {accuracy_gap:.4f} ({accuracy_gap/accuracy*100:.2f}%)")
print(f"F1-Score Drop: {f1_gap:.4f} ({f1_gap/f1*100:.2f}%)")

metrics = ['Accuracy', 'F1-Score']
test_set_1_values = [accuracy, f1]
test_set_2_values = [accuracy_test_v2, f1_test_v2]
gaps = [accuracy_gap, f1_gap]

x = np.arange(len(metrics)) 
width = 0.35  

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, test_set_1_values, width, label='Test Set 1')
rects2 = ax.bar(x + width/2, test_set_2_values, width, label='Test Set 2')


ax.set_ylabel('Scores')
ax.set_title('Comparison of Performance Metrics Between Test Sets')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()