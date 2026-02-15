"""
ML Assignment 2 - Streamlit Web Application
Interactive Classification Model Comparison Dashboard

Features:
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split

# ML Models (for retraining if needed)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Try importing xgboost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Classification Model Comparison Dashboard")
st.markdown("**Machine Learning Assignment 2 - BITS Pilani WILP**")
st.markdown("---")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_model(model_path):
    """Load a saved pickle model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all 6 evaluation metrics."""
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC Score": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC Score": round(matthews_corrcoef(y_true, y_pred), 4)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Bad (0)', 'Good (1)'],
                yticklabels=['Bad (0)', 'Good (1)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    return fig

def get_models_dict():
    """Return dictionary of model objects."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    if xgb_available:
        models["XGBoost (Ensemble)"] = XGBClassifier(
            n_estimators=100, random_state=42,
            use_label_encoder=False, eval_metric='logloss'
        )
    return models

# ============================================================
# SIDEBAR - Dataset Upload
# ============================================================
st.sidebar.header("📂 Data Upload")
st.sidebar.markdown("Upload a CSV file with test data. The file should have the same features as the Wine Quality dataset and a `target` column.")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Load data
data_loaded = False

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success(f"File uploaded! Shape: {data.shape}")
        data_loaded = True
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
else:
    st.sidebar.info("No file uploaded. Using default Wine Quality dataset.")
    # Load default dataset
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        data = pd.read_csv(url, sep=';')
        data['target'] = (data['quality'] >= 7).astype(int)
        data = data.drop('quality', axis=1)
        data_loaded = True
    except:
        st.error("Could not load default dataset. Please upload a CSV file.")

if data_loaded:
    # ============================================================
    # SIDEBAR - Model Selection
    # ============================================================
    st.sidebar.header("🔧 Model Selection")

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
    ]
    if xgb_available:
        model_names.append("XGBoost (Ensemble)")

    selected_model = st.sidebar.selectbox(
        "Select a classification model:",
        model_names
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Compare all models** by checking the box below:")
    compare_all = st.sidebar.checkbox("Show all models comparison", value=True)

    # ============================================================
    # MAIN CONTENT
    # ============================================================

    # Dataset Overview
    st.header("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", data.shape[0])
    col2.metric("Features", data.shape[1] - 1)
    col3.metric("Target Classes", data['target'].nunique())

    with st.expander("View Dataset Sample"):
        st.dataframe(data.head(10))
        st.write(f"**Shape:** {data.shape}")

    with st.expander("Dataset Statistics"):
        st.dataframe(data.describe())

    st.markdown("---")

    # Prepare data for modeling
    if 'target' in data.columns:
        X = data.drop('target', axis=1)
        y = data['target']
    else:
        st.error("Dataset must have a 'target' column. Please check your CSV file.")
        st.stop()

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================================================
    # SINGLE MODEL RESULTS
    # ============================================================
    st.header(f"🎯 Selected Model: {selected_model}")

    models = get_models_dict()
    model = models[selected_model]

    # Train the selected model
    with st.spinner(f"Training {selected_model}..."):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # Display metrics in columns
    st.subheader("📈 Evaluation Metrics")
    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
    mcol1.metric("Accuracy", metrics["Accuracy"])
    mcol2.metric("AUC Score", metrics["AUC Score"])
    mcol3.metric("Precision", metrics["Precision"])
    mcol4.metric("Recall", metrics["Recall"])
    mcol5.metric("F1 Score", metrics["F1 Score"])
    mcol6.metric("MCC Score", metrics["MCC Score"])

    # Confusion Matrix and Classification Report side by side
    st.subheader("📊 Confusion Matrix & Classification Report")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        fig = plot_confusion_matrix(y_test, y_pred, selected_model)
        st.pyplot(fig)

    with viz_col2:
        st.text("Classification Report:")
        report = classification_report(y_test, y_pred,
                                       target_names=['Bad Wine (0)', 'Good Wine (1)'])
        st.code(report)

    st.markdown("---")

    # ============================================================
    # ALL MODELS COMPARISON
    # ============================================================
    if compare_all:
        st.header("📋 All Models Comparison")

        all_results = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        all_models = get_models_dict()
        total = len(all_models)

        for idx, (name, mdl) in enumerate(all_models.items()):
            status_text.text(f"Training {name}...")
            mdl.fit(X_train_scaled, y_train)
            pred = mdl.predict(X_test_scaled)
            prob = mdl.predict_proba(X_test_scaled)[:, 1]
            all_results[name] = calculate_metrics(y_test, pred, prob)
            progress_bar.progress((idx + 1) / total)

        status_text.text("All models trained!")

        # Comparison table
        comparison_df = pd.DataFrame(all_results).T
        comparison_df.index.name = "ML Model Name"
        st.dataframe(comparison_df, use_container_width=True)

        # Bar chart comparison
        st.subheader("📊 Visual Comparison")
        fig, ax = plt.subplots(figsize=(12, 5))
        comparison_df.plot(kind='bar', ax=ax, edgecolor='black')
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Best model
        best_model_name = comparison_df['Accuracy'].idxmax()
        best_acc = comparison_df['Accuracy'].max()
        st.success(f"🏆 **Best Model by Accuracy:** {best_model_name} ({best_acc})")

    st.markdown("---")
    st.markdown("*ML Assignment 2 - BITS Pilani WILP | Machine Learning Course*")
