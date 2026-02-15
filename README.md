# ML Assignment 2 - Classification Model Comparison

## Problem Statement

The goal of this project is to build and compare multiple machine learning classification models on the **Wine Quality (Red Wine)** dataset. We classify wines as **Good Quality (rating >= 7)** or **Bad Quality (rating < 7)** based on their physicochemical properties. The project implements 6 different classification algorithms, evaluates them using 6 standard metrics, and deploys an interactive Streamlit web application for demonstration.

## Dataset Description

- **Dataset Name:** Wine Quality (Red Wine)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Total Samples:** 1,599
- **Number of Features:** 11 (all numerical)
- **Target Variable:** Binary classification — Good (quality >= 7) → 1, Bad (quality < 7) → 0
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
- **Train/Test Split:** 80/20 (stratified)
- **Missing Values:** None

## Models Used

All 6 classification models were implemented on the same dataset. Below is the comparison table with evaluation metrics:

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8937 | 0.8413 | 0.6667 | 0.4000 | 0.5000 | 0.4753 |
| Decision Tree | 0.8813 | 0.7476 | 0.5455 | 0.4000 | 0.4615 | 0.4115 |
| kNN | 0.8750 | 0.8380 | 0.5000 | 0.3667 | 0.4231 | 0.3704 |
| Naive Bayes | 0.8375 | 0.8262 | 0.3846 | 0.5000 | 0.4348 | 0.3647 |
| Random Forest (Ensemble) | 0.9125 | 0.8980 | 0.7143 | 0.5000 | 0.5882 | 0.5614 |
| XGBoost (Ensemble) | 0.9062 | 0.8800 | 0.7000 | 0.4667 | 0.5600 | 0.5288 |

> **Note:** Exact metric values may vary slightly depending on the execution environment and random seed behavior.

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression provides a solid baseline with high accuracy (0.89) and good AUC (0.84). It performs well on this dataset because the decision boundary between good and bad wines can be approximated linearly. However, its recall is moderate (0.40), meaning it misses some good wines. |
| Decision Tree | Decision Tree achieves decent accuracy (0.88) but has the lowest AUC (0.75) among all models, indicating it struggles with ranking predictions. It tends to overfit on specific patterns in the training data, even with max_depth=5 regularization. It captures non-linear relationships but lacks generalization. |
| kNN | K-Nearest Neighbors shows moderate performance across all metrics. Its accuracy (0.875) is comparable to Decision Tree. The distance-based approach is sensitive to feature scaling (handled via StandardScaler). Its lower precision and recall suggest that the decision boundary in high-dimensional space is not clearly defined by nearest neighbors. |
| Naive Bayes | Naive Bayes has the lowest accuracy (0.84) but interestingly achieves the highest recall (0.50) among non-ensemble models, meaning it identifies more actual good wines. The naive independence assumption between features limits its precision. It works as a quick probabilistic baseline but is not ideal when features are correlated, as is the case with wine chemistry. |
| Random Forest (Ensemble) | Random Forest achieves the best overall performance with the highest accuracy (0.91), AUC (0.90), precision (0.71), F1 (0.59), and MCC (0.56). As an ensemble of decision trees, it reduces overfitting through bagging and feature randomization. It handles the imbalanced dataset better than individual models and captures complex non-linear interactions between features. |
| XGBoost (Ensemble) | XGBoost is the second-best model with accuracy of 0.91 and AUC of 0.88. Its gradient boosting approach sequentially corrects errors from previous trees, resulting in strong predictive performance. It is slightly behind Random Forest on this specific dataset, likely because the dataset size is moderate and Random Forest's bagging approach provides sufficient regularization. |

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/zaifsitemt/ml-assignment-2.git
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

🔗 **Streamlit App:** [Click here to open the app](https://ml-assignment-2-2025aa05222-prog.streamlit.app/)

## Technologies Used

- Python 3.9+
- Scikit-learn
- XGBoost
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
