# 📉 Customer Churn Prediction

Predicting which telecom customers are likely to churn using machine learning, enabling the business to take proactive retention actions before losing the customer.

---

## 📌 Problem Statement

Customer churn is one of the most costly problems for telecom companies. Acquiring a new customer costs 5–10x more than retaining an existing one. This project builds a machine learning pipeline that identifies at-risk customers early, so the business can intervene with offers or support before they leave.

---

## 📂 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customers × 21 features
- **Target:** Churn (Yes / No)
- **Class Distribution:** 73.5% No Churn → 5,174 customers, 26.5% Churn → 1,869 customers
- **Features include:** tenure, MonthlyCharges, TotalCharges, Contract type, PaymentMethod, InternetService, and more

---

## 🔍 Exploratory Data Analysis — Key Findings

- **New customers churn most** — churn rate is highest in the first 1–5 months of tenure, then drops sharply
- **Month-to-month contract customers** churn at nearly 50% — far higher than 1-year or 2-year contract holders
- **Higher monthly charges correlate with churn** — churners had a median monthly charge of ~₹80 vs ~₹65 for loyal customers
- **Long-term customers almost never churn** — customers past 24 months are highly stable

---

## 🛠️ Approach

### 1. Data Cleaning
- Discovered `TotalCharges` was incorrectly stored as `object` (string) dtype despite being numeric
- Converted to float and found **11 missing values** — filled with median
- Dropped `customerID` as it carries no predictive value

### 2. Feature Engineering
Created 2 new features from domain logic:
- **`AvgMonthlySpend`** = TotalCharges / (tenure + 1) — captures spending pattern over time
- **`IsLongTermCustomer`** = 1 if tenure > 24 months — captures customer stability

Both features ranked in the **top 10 most important features** out of 32 total, confirming they added real predictive value.

### 3. Encoding
- Binary columns (Partner, Dependents, PhoneService, PaperlessBilling) encoded as 0/1
- Categorical columns (Contract, PaymentMethod, InternetService, etc.) one-hot encoded
- Final feature count: **32 features**

### 4. Class Imbalance — SMOTE
Applied **SMOTE (Synthetic Minority Oversampling Technique)** exclusively on training data:
- Before SMOTE: {No Churn: 4139, Churn: 1495}
- After SMOTE: {No Churn: 4139, Churn: 4139}

> ⚠️ SMOTE was applied only after train/test split to prevent synthetic samples from leaking into the test set.

### 5. Model Training
Trained and compared 4 models on the SMOTE-balanced training data, evaluated on real unseen test data (1,409 customers):

---

## 📊 Results

| Model | F1 | AUC | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.613** | **0.846** | 0.505 | **0.781** |
| XGBoost | 0.613 | 0.839 | 0.575 | 0.655 |
| Decision Tree | 0.610 | 0.812 | 0.530 | 0.719 |
| Random Forest | 0.573 | 0.825 | 0.562 | 0.586 |

### ✅ Best Model: Logistic Regression

**Why Logistic Regression over XGBoost?**
Despite tying on F1 (0.613), Logistic Regression achieved higher AUC (0.846 vs 0.839) and significantly higher Recall (0.781 vs 0.655). On this dataset — relatively small and clean after preprocessing — the simpler model generalizes better.

**Why prioritize Recall over Precision?**
Missing an actual churner (False Negative) is more costly to the business than a false alarm (False Positive). A false alarm means an unnecessary retention offer; a missed churner means losing the customer entirely.

---

## 🔢 Confusion Matrix (Logistic Regression)

```
                  Predicted: No Churn    Predicted: Churn
Actual: No Churn        749                   286
Actual: Churn            82                   292
```

- ✅ **292 churners correctly identified** out of 374 total
- ❌ **82 churners missed** (false negatives)
- ⚠️ **286 false alarms** (loyal customers flagged as churners)

**Business impact:** The model successfully flags 78% of at-risk customers, giving the retention team a targeted list to act on instead of reaching out to all 7,043 customers blindly.

---

## 🏆 Top Features Driving Churn

| Rank | Feature | Type |
|---|---|---|
| 1 | tenure | Original |
| 2 | TotalCharges | Original |
| 3 | MonthlyCharges | Original |
| 4 | AvgMonthlySpend | **Engineered** |
| 5 | Contract_Two year | Original (encoded) |
| 6 | PaymentMethod_Electronic check | Original (encoded) |
| 7 | InternetService_Fiber optic | Original (encoded) |
| 8 | PaperlessBilling | Original |
| 9 | IsLongTermCustomer | **Engineered** |
| 10 | OnlineSecurity_Yes | Original (encoded) |

> Both engineered features (AvgMonthlySpend and IsLongTermCustomer) appear in the top 10, confirming that feature engineering added meaningful signal beyond the raw data.

---

## 💡 Business Recommendations

Based on model findings:
1. **Focus retention efforts on new customers (0–5 months tenure)** — highest churn risk window
2. **Offer incentives to switch from month-to-month to annual contracts** — single biggest churn reducer
3. **Flag electronic check users for proactive outreach** — payment method is a top churn signal
4. **Review Fiber Optic pricing** — Fiber optic users churn more, possibly due to cost vs value perception

---

## 🧰 Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Environment | Google Colab |

---

## 🚀 How to Run

1. Open `Customer_Churn_Prediction.ipynb` in [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Run All**
3. All dependencies are pre-installed in Colab

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction.ipynb   # Main notebook (full pipeline)
└── README.md                         # Project documentation
```

---

## 👤 Author

**Davinder Kaur**
B.Tech CSE | The ICFAI University, Jaipur
📧 kaur.btech2023@iujaipur.edu.in | 💻 [GitHub](https://github.com/davii1207)
