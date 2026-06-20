  # 💳 Credit Card Fraud Detection using Decision Tree Classifier

Detecting fraudulent credit card transactions on a highly imbalanced, real-world dataset using a Decision Tree Classifier (with a Random Forest comparison), built with Python and scikit-learn.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Details](#-model-details)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Future Improvements](#-future-improvements)
- [Authors](#-authors)
- [Acknowledgments](#-acknowledgments)

---

## 🔎 Overview

Credit card fraud costs banks and customers billions of dollars every year, and fraudulent transactions make up a tiny fraction of all transactions — making them very hard to catch. This project builds a machine learning pipeline that flags fraudulent transactions while handling extreme class imbalance (only ~0.17% of transactions are fraud).

The pipeline covers data cleaning, robust scaling, exploratory data analysis, stratified train/test splitting, and classification using a **Decision Tree** (primary model) and a **Random Forest** (bonus comparison model), evaluated with metrics suited for imbalanced data such as Recall and ROC-AUC.

## 🎯 Problem Statement

Out of 284,807 transactions, only 492 are fraudulent. A naive model can score 99.8% accuracy by predicting "not fraud" every time — while catching zero fraud. The real goal is building a model that is **sensitive enough to catch fraud** without overwhelming the system with false alarms on legitimate customers.

## 📊 Dataset

- **Source:** [Credit Card Fraud Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions:** 284,807 (European cardholders, September 2013)
- **Fraud cases:** 492 (≈0.17% of the data)
- **Features:** 31 columns
  - `V1`–`V28`: anonymized/PCA-transformed features (real names hidden for privacy)
  - `Time`: seconds elapsed since the first transaction
  - `Amount`: transaction amount
  - `Class`: target label (`0` = Normal, `1` = Fraud)

> ⚠️ The dataset file (`creditcard.csv`, ~150 MB) is **not included** in this repo due to GitHub's file size limits. Download it from Kaggle (link above) and place it in the project root before running the notebook.

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Environment | Jupyter Notebook |

## 🔁 Project Workflow

1. **Data Loading & Cleaning** — Load the CSV, check shape, check for nulls and duplicates, remove duplicate rows.
2. **Preprocessing & Scaling** — Apply `RobustScaler` to `Amount` and `Time` to neutralize the effect of outliers common in financial data.
3. **Exploratory Data Analysis (EDA)** — Visualize class imbalance and feature relationships (see below).
4. **Stratified Train/Test Split** — 70/30 split using `stratify=y` so both sets preserve the same fraud ratio.
5. **Model Training** — Train a `DecisionTreeClassifier` (entropy criterion, `max_depth=5` to avoid overfitting).
6. **Bonus Model** — Train a `RandomForestClassifier` (`n_estimators=50`) for comparison.
7. **Evaluation** — Compare both models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## 📈 Exploratory Data Analysis

Five visualizations were built to understand the imbalanced data:

1. **Log-Scale Count Plot** — highlights how rare fraud cases are.
2. **Correlation Heatmap** — finds which `V` features correlate most strongly with fraud.
3. **Violin Plot (V14)** — shows the distribution spread of a key feature across classes.
4. **Density Plot (V12)** — compares the distribution of Normal vs Fraud transactions.
5. **3D Scatter Plot (V1, V2, V3)** — visualizes how fraud points cluster in 3D feature space.

## 🤖 Model Details

**Decision Tree Classifier**
- Criterion: `entropy` (Information Gain)
- Max depth: `5` (to prevent overfitting/memorization)
- Strength: highly interpretable — produces clear, explainable rules
- Weakness: a single tree can overfit if left unconstrained

**Random Forest Classifier (Bonus)**
- 50 estimators (`n_estimators=50`)
- Combines multiple trees for a more stable, generalized prediction

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (Main) | 0.999 | 0.85 | 0.75 | 0.80 | 0.88 |
| Random Forest (Bonus) | 0.999 | 0.92 | 0.81 | 0.86 | 0.94 |

**Key takeaway:** Both models achieve ~99.9% accuracy (expected, given the imbalance), but Random Forest shows meaningfully better Recall and ROC-AUC — meaning it catches more actual fraud cases and separates the two classes more reliably.

## 📂 Project Structure

```
credit-card-fraud-detection/
├── Fraud_Detection.ipynb     # Main notebook: EDA, preprocessing, models, evaluation
├── README.md                 # Project documentation (this file)
```

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Get `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root folder.

**4. Run the notebook**
```bash
jupyter notebook Fraud_Detection.ipynb
```

## 🔮 Future Improvements

- Apply resampling techniques (SMOTE, undersampling) to further address class imbalance
- Tune hyperparameters with `GridSearchCV` for both models
- Try gradient boosting models (XGBoost, LightGBM) for comparison
- Deploy the model behind a simple API for real-time scoring

## 👥 Authors

- **Zaid Qasim**

## 🙏 Acknowledgments

- Dataset provided by [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via Kaggle
- Built as part of an Artificial Intelligence Lab project
