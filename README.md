# ml-credit-scoring-model
A machine learning model to predict credit default using Random Forest
# Credit Risk Prediction Model

A machine learning project to predict an individual's creditworthiness using past financial data. This model leverages classification algorithms to assess the probability of loan default.

## 🎯 Objective
Predict whether an individual will default on a loan (binary classification) using features like income, debt, payment history, and credit length.

## 📊 Dataset
The model is trained on the ["Give Me Some Credit" dataset from Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-Learn, Matplotlib, Seaborn
- **Environment:** Google Colab

## 📁 Project Structure
credit-risk-prediction/
├── credit_scoring_model.ipynb # Main Jupyter Notebook
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)

## 🚀 How to Run
1.  Clone this repository: `git clone <your-repository-url>`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Open and run the `credit_scoring_model.ipynb` notebook in Jupyter or Google Colab.

## ⚙️ Methodology
1.  **Data Preprocessing:** Handled missing values and encoded categorical variables.
2.  **Class Imbalance:** Addressed using **SMOTE** (Synthetic Minority Over-sampling Technique).
3.  **Model Training:** Evaluated and compared **Logistic Regression** and **Random Forest** algorithms.
4.  **Evaluation:** Assessed model performance using Precision, Recall, F1-Score, and **ROC-AUC**.

## 📈 Results
The **Random Forest** model significantly outperformed the baseline, achieving an excellent **ROC-AUC score of 0.9289**.

| Model                | ROC-AUC Score |
| -------------------- | ------------- |
| Logistic Regression  | 0.8214        |
| **Random Forest**    | **0.9289**    |

## 🔍 Feature Importance
The most important features for predicting default were:
1.  `loan_int_rate` (Interest Rate)
2.  `person_income` (Income)
3.  `cb_person_cred_hist_length` (Credit History Length)
4.  `loan_amnt` (Loan Amount)
5.  `person_age` (Age)
