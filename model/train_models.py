"""
ML Assignment 2 - Classification Models Training Script
Dataset: Wine Quality (Red Wine) from UCI Machine Learning Repository
Task: Binary Classification - Predict if wine quality is Good (>=7) or Bad (<7)

This script trains 6 classification models and saves them as pickle files.
Run this script first before running the Streamlit app.
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


print("=" * 60)
print("ML ASSIGNMENT 2 - CLASSIFICATION MODELS")
print("=" * 60)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
try:
    df = pd.read_csv(url, sep=';')
    print(f"\nDataset loaded from UCI repository")
except:
   
    df = pd.read_csv("winequality-red.csv", sep=';')
    print(f"\nDataset loaded from local file")

print(f"Shape: {df.shape}")
print(f"Features: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())


print(f"\nDataset Statistics:")
print(df.describe())


print(f"\nMissing values:\n{df.isnull().sum()}")


df['target'] = (df['quality'] >= 7).astype(int)

print(f"\nTarget Distribution:")
print(df['target'].value_counts())
print(f"Good wine (>=7): {df['target'].sum()}")
print(f"Bad wine (<7): {(df['target'] == 0).sum()}")


X = df.drop(['quality', 'target'], axis=1)
y = df['target']

print(f"\nFeature columns ({X.shape[1]} features): {X.columns.tolist()}")
print(f"Total samples: {X.shape[0]}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("\nScaler saved.")

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['target'] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("Test data saved to test_data.csv")


with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost (Ensemble)": XGBClassifier(n_estimators=100, random_state=42,
                                         use_label_encoder=False,
                                         eval_metric='logloss')
}

results = {}

print("\n" + "=" * 60)
print("TRAINING AND EVALUATING MODELS")
print("=" * 60)

for name, model in models.items():
    print(f"\n--- {name} ---")


    model.fit(X_train_scaled, y_train)


    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

 
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    }

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")


    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(f"model/{safe_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to model/{safe_name}.pkl")


print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model Name"
print(results_df.to_string())


results_df.to_csv("model/results.csv")
print("\nResults saved to model/results.csv")


with open("model/results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\n" + "=" * 60)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 60)
