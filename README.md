# ML Assignment 2 - Classification Model Comparison

## Problem Statement

The goal of this project is to build and compare multiple machine learning classification models on the **Wine Quality (Red Wine)** dataset. We classify wines as **Good Quality (rating >= 7)** or **Bad Quality (rating < 7)** based on their physicochemical properties. The project implements 6 different classification algorithms, evaluates them using 6 standard metrics, and deploys an interactive Streamlit web application for demonstration.

## Dataset Description

- **Dataset Name:** Wine Quality (Red Wine)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Total Samples:** 1,599
- **Number of Features:** 11 (all numerical)
- **Target Variable:** Binary classification — Good (quality >= 7) → 1, Bad (quality < 7) → 0
- **Class Distribution:** Good wine: 217, Bad wine: 1382
- **Features:**
  1. Fixed acidity
  2. Volatile acidity
  3. Citric acid
  4. Residual sugar
  5. Chlorides
  6. Free sulfur dioxide
  7. Total sulfur dioxide
  8. Density
  9. pH
  10. Sulphates
  11. Alcohol
- **Train/Test Split:** 80/20 (stratified) — Train: 1279, Test: 320
- **Missing Values:** None

## Models Used

All 6 classification models were implemented on the same dataset. Below is the comparison table with evaluation metrics:

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8938 | 0.8804 | 0.6957 | 0.3721 | 0.4848 | 0.4580 |
| Decision Tree | 0.9187 | 0.8670 | 0.7297 | 0.6279 | 0.6750 | 0.6312 |
| kNN | 0.8938 | 0.8237 | 0.6667 | 0.4186 | 0.5143 | 0.4738 |
| Naive Bayes | 0.8594 | 0.8517 | 0.4844 | 0.7209 | 0.5794 | 0.5131 |
| Random Forest (Ensemble) | 0.9375 | 0.9547 | 0.9259 | 0.5814 | 0.7143 | 0.7045 |
| XGBoost (Ensemble) | 0.9406 | 0.9422 | 0.8750 | 0.6512 | 0.7467 | 0.7239 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression provides a solid baseline with accuracy of 0.8938 and AUC of 0.8804. It performs well because the decision boundary between good and bad wines can be approximated linearly. However, recall is low (0.3721) meaning it misses many good wines. |
| Decision Tree | Decision Tree achieves good accuracy (0.9187) and strong recall (0.6279) compared to simpler models. It captures non-linear patterns in the data effectively. AUC of 0.8670 shows reasonable ranking ability, though it can overfit on training data. |
| kNN | K-Nearest Neighbors shows moderate performance with accuracy 0.8938 and AUC 0.8237. The distance-based approach is sensitive to feature scaling (handled via StandardScaler). Its recall (0.4186) suggests the decision boundary is not well-defined by nearest neighbors. |
| Naive Bayes | Naive Bayes achieves the highest recall (0.7209) among non-ensemble models, correctly identifying more good wines. However, its precision is lowest (0.4844) due to the naive independence assumption between correlated wine chemistry features. Good as a probabilistic baseline. |
| Random Forest (Ensemble) | Random Forest achieves excellent accuracy (0.9375) and the highest AUC (0.9547) with outstanding precision (0.9259). It reduces overfitting through bagging and feature randomization. MCC of 0.7045 confirms strong balanced performance on this imbalanced dataset. |
| XGBoost (Ensemble) | XGBoost is the best model overall with highest accuracy (0.9406), best recall among high-precision models (0.6512), highest F1 (0.7467) and highest MCC (0.7239). Its gradient boosting approach sequentially corrects errors, giving the best balance of precision and recall. |

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/2025aa05222-prog/ml-assignment-2.git
cd ml-assignment-2

# Install dependencies
pip install -r requirements.txt

# (Optional) Train models and generate results
cd model
python train_models.py
cd ..

# Run the Streamlit app
streamlit run app.py
```

## Project Structure

```
ml-assignment-2/
│-- app.py                    # Streamlit web application
│-- requirements.txt          # Python dependencies
│-- README.md                 # Project documentation
│-- test_data.csv             # Sample test data for upload
│-- model/
│   │-- train_models.py       # Model training script
│   │-- scaler.pkl            # Saved StandardScaler
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- xgboost_ensemble.pkl
│   │-- results.csv           # Comparison table
│   │-- results.pkl           # Saved results dictionary
│   │-- feature_names.pkl     # Feature names list
```

## Live App

🔗 **Streamlit App:** [https://ml-assignment-2-2025aa05222-prog.streamlit.app/]

## Technologies Used

- Python 3.9+
- Scikit-learn
- XGBoost
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
