import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="BankRisk AI - CIBIL Predictor", page_icon="🏦", layout="wide")


# 2. Load Model Efficiently
@st.cache_resource
def load_model():
    # Ensure best_model.joblib is in the same directory as this script
    return joblib.load('best_model.joblib')


model = load_model()

# 3. App Header
st.title("🏦 BankRisk AI: CIBIL Score Predictor")
st.markdown("Predict applicant credit scores and analyze risk factors using machine learning.")
st.divider()

# 4. User Input Section
st.subheader("Applicant Financial Profile")
st.markdown("Enter the applicant's details below based on the 9 key predictive features.")

with st.form("applicant_form"):
    st.markdown("### 👤 Demographics & Flags")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (AGE)", min_value=18, max_value=100, value=30)
    with col2:
        approved_flag_p1 = st.selectbox("Tier 1 Approval Flag (Approved_Flag_P1)", options=[0, 1],
                                        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    with col3:
        approved_flag_p4 = st.selectbox("Tier 4 Approval Flag (Approved_Flag_P4)", options=[0, 1],
                                        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    st.markdown("### 💳 Standard Payment History")
    col4, col5, col6 = st.columns(3)

    with col4:
        num_std = st.number_input("Total Standard Payments (num_std)", min_value=0, value=15)
    with col5:
        num_std_12mts = st.number_input("Standard Payments Last 12M (num_std_12mts)", min_value=0, value=10)
    with col6:
        num_std_6mts = st.number_input("Standard Payments Last 6M (num_std_6mts)", min_value=0, value=5)

    st.markdown("### 📊 Enquiries & Tradelines Ratios")
    col7, col8, col9 = st.columns(3)

    with col7:
        pct_pl_enq_l6m_ever = st.number_input("% PL Enquiries: 6M vs Ever (pct_PL_enq_L6m_of_ever)", min_value=0.0,
                                              value=0.2, step=0.01)
    with col8:
        pct_pl_enq_l6m_l12m = st.number_input("% PL Enquiries: 6M vs 12M (pct_PL_enq_L6m_of_L12m)", min_value=0.0,
                                              value=0.5, step=0.01)
    with col9:
        pct_tl_open_l12m = st.number_input("% Tradelines Opened Last 12M (pct_tl_open_L12M)", min_value=0.0, value=0.1,
                                           step=0.01)

    submitted = st.form_submit_button("Predict CIBIL Score & Analyze Risk")

# 5. Prediction and SHAP Explainability Logic
if submitted:
    # Compile the inputs into a DataFrame using the exact column order from your notebook
    input_data = {
        'Approved_Flag_P1': [approved_flag_p1],
        'Approved_Flag_P4': [approved_flag_p4],
        'num_std_12mts': [num_std_12mts],
        'num_std': [num_std],
        'num_std_6mts': [num_std_6mts],
        'pct_PL_enq_L6m_of_ever': [pct_pl_enq_l6m_ever],
        'pct_PL_enq_L6m_of_L12m': [pct_pl_enq_l6m_l12m],
        'pct_tl_open_L12M': [pct_tl_open_l12m],
        'AGE': [age]
    }

    input_df = pd.DataFrame(input_data)

    with st.spinner("Analyzing risk profile..."):
        # Predict Score
        prediction = model.predict(input_df)[0]

        # Display Prediction
        st.divider()
        st.subheader("🎯 Prediction Results")

        # Color code the score based on standard CIBIL brackets
        if prediction >= 750:
            color = "#28a745"  # Green
            status = "EXCELLENT"
        elif prediction >= 700:
            color = "#8bc34a"  # Light Green
            status = "GOOD"
        elif prediction >= 650:
            color = "#ffc107"  # Yellow/Orange
            status = "FAIR"
        else:
            color = "#dc3545"  # Red
            status = "POOR"

        st.markdown(f"### Predicted CIBIL Score: <span style='color:{color}'>{int(prediction)}</span> ({status})",
                    unsafe_allow_html=True)

        # --- SHAP Explainability ---
        st.divider()
        st.subheader("🔍 Risk Factor Analysis (SHAP)")
        st.markdown(
            "This waterfall chart explains how specific financial behaviors pushed the applicant's score up or down from the baseline average.")

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)

        # Create the SHAP Waterfall Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # shap_values[0] because we are only explaining the single row of input
        shap.plots.waterfall(shap_values[0], show=False)

        # Adjust layout so labels don't get cut off in Streamlit
        plt.tight_layout()

        # Display the plot
        st.pyplot(fig)

        # Clear the plot to prevent memory leaks in the app
        plt.clf()