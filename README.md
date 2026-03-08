# CIBIL Score Prediction Pipeline

## 📌 The Real-World Problem
In the financial sector, assessing credit risk is a foundational challenge. Banks and lending institutions must accurately predict an applicant's creditworthiness—represented by their CIBIL score—to make informed decisions regarding loan approvals, credit limits, and interest rates. 

Relying purely on internal banking history or solely on external credit bureau reports often provides an incomplete picture of human financial behavior. The problem this project solves is accurately predicting a continuous credit score by synthesizing dual data sources, allowing for highly reliable risk assessment in a corporate analytics environment.

## 🎯 Project Scope
This project implements an end-to-end machine learning pipeline that merges internal bank records with external CIBIL datasets to predict an individual's `Credit_Score`. 

**In Scope:**
* Merging and harmonizing disparate financial datasets.
* Handling legacy banking data artifacts (e.g., `-99999` missing value placeholders).
* Training a predictive regression model using gradient boosting.
* Hyperparameter tuning for optimal bias-variance tradeoff.
* Serializing the final production-ready model for deployment.

**Out of Scope:**
* Real-time data streaming (the model is trained in a batch-processing environment).
* Automated loan approval logic (the model outputs the score; business logic dictates the threshold).

## ⚙️ Methodology & Steps

1. **Data Integration:** * Merged `Internal_Bank_Dataset.xlsx` and `External_Cibil_Dataset.xlsx` to create a holistic view of the applicant's financial footprint.
2. **Data Preprocessing (`Data-Preprocessing.ipynb`):**
   * Handled extreme placeholder values indicative of missing data.
   * Split the data into training and testing sets, ensuring data leakage prevention.
   * Exported clean splits (`x_train.csv`, `y_train.csv`, etc.) for modular model training.
3. **Model Training (`model-training.ipynb`):**
   * Implemented `XGBRegressor` to capture non-linear relationships and complex interactions in the tabular data.
4. **Hyperparameter Tuning:**
   * Utilized `GridSearchCV` to exhaustively search the optimal parameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`).
5. **Serialization:**
   * Exported the best estimator as `best_model.joblib` for future inference and application deployment.

## 📊 Model Performance & Metrics
The model was evaluated against a target `Credit_Score` ranging from 469 to 811 (mean: ~679). The metrics indicate a highly consistent and reliable predictive capability:

* **$R^2$ Score:** `0.7402` 
  * *Interpretation:* The model successfully explains 74% of the variance in the dataset, a strong indicator of capturing underlying financial risk signals without overfitting.
* **Root Mean Squared Error (RMSE):** `10.42`
  * *Interpretation:* Penalizes larger errors, confirming that the model rarely makes extreme miscalculations (e.g., predicting 500 when the actual is 800).
* **Mean Absolute Error (MAE):** `8.07`
  * *Interpretation:* On average, the model's predictions are only ~8 points away from the actual CIBIL score. In a practical banking context, this tight margin of error provides highly reliable decision support.

## 🛠️ Technologies Used
* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Joblib
* **Environment:** Jupyter Notebooks

## 🚀 Future Enhancements
* **Explainable AI (XAI):** Integrate SHAP values to explain feature importance (e.g., how recent delinquencies impact the score vs. credit utilization).
* **Application Deployment:** Wrap the inference pipeline in a Streamlit web application to allow users to input financial parameters and receive a real-time CIBIL score prediction.

## 👨‍💻 Author
**Mithul Krishna Suresh** B.Tech in Computer Science and Engineering (2nd Year)  
National Institute of Technology, Bhopal (MANIT)  
Roll Number: 24112011206