# 🏦 BankRisk AI: CIBIL Score Prediction & Explainability Pipeline

## 📌 The Real-World Problem
In the financial sector, assessing credit risk is a foundational challenge. Banks and lending institutions must accurately predict an applicant's creditworthiness—represented by their CIBIL score—to make informed decisions regarding loan approvals, credit limits, and interest rates. 

Relying purely on internal banking history or solely on external credit bureau reports often provides an incomplete picture of human financial behavior. Furthermore, modern financial regulations require **Explainable AI (XAI)**; bank employees cannot blindly trust a black-box model—they must know *why* a score was assigned.

## 🎯 Project Scope & Architecture
This project implements an end-to-end machine learning and deployment pipeline that merges internal bank records with external CIBIL datasets to predict an individual's `Credit_Score`, complete with a front-end UI and feature explainability.

**Core Components:**
1. **Data Pipeline:** Merging datasets, handling legacy banking data artifacts (e.g., `-99999` missing value placeholders), and one-hot encoding categorical variables.
2. **Feature Selection:** Extracting the top 9 highly correlated predictive features focusing on behavioral velocity rather than just static demographics.
3. **Model Training:** Hyperparameter-tuned `XGBRegressor` to capture non-linear relationships.
4. **Interactive UI:** A Streamlit web application for real-time inference.
5. **Explainable AI:** SHAP (SHapley Additive exPlanations) integration to visually break down the risk factors for every individual prediction.

## 🧠 The 9 Key Predictive Features
Instead of using dozens of noisy variables, the model was optimized to use 9 highly predictive features across three risk categories:

* **Internal Risk Flags:** `Approved_Flag_P1` (Prime Tier) & `Approved_Flag_P4` (Subprime Tier).
* **Demographics:** `AGE`.
* **Standard Payment Velocity:** `num_std` (Lifetime), `num_std_12mts` (Last 12 Months), `num_std_6mts` (Last 6 Months).
* **Credit-Seeking Behavior:** `pct_PL_enq_L6m_of_ever` (Recent Personal Loan Enquiries vs Lifetime), `pct_PL_enq_L6m_of_L12m`, and `pct_tl_open_L12M` (Recent Tradelines Opened).

## 📊 Model Performance & Metrics
The model was evaluated against a target `Credit_Score` ranging from 469 to 811 (mean: ~679). 

* **$R^2$ Score:** `0.7402` (Successfully explains 74% of the variance in human financial behavior, demonstrating a strong capture of risk signals).
* **Root Mean Squared Error (RMSE):** `10.42` (Penalizes larger errors, confirming the model rarely makes extreme miscalculations).
* **Mean Absolute Error (MAE):** `8.07` (On average, the model's predictions are only ~8 points away from the actual CIBIL score, providing highly reliable decision support).

## 🛠️ Technologies Used
* **Machine Learning:** Scikit-Learn, XGBoost (`XGBRegressor`)
* **Data Processing:** Pandas, NumPy
* **Explainable AI:** SHAP
* **Web Deployment:** Streamlit, Matplotlib
* **Model Serialization:** Joblib

## 🚀 How to Run the Application

1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit shap matplotlib joblib

## 👨‍💻 Author
**Mithul Krishna Suresh** B.Tech in Computer Science and Engineering (2nd Year)  
National Institute of Technology, Bhopal (MANIT)  
Roll Number: 24112011206