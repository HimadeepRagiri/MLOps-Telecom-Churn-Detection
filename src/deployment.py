import streamlit as st
import joblib
import os
import pandas as pd
from loguru import logger
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all preprocessing and feature engineering functions for inference
from src.preprocessing import (
    convert_object_to_numeric, encode_binary_columns, handle_missing_values,
    handle_duplicates, standardize_formats, handle_outliers,
    drop_irrelevant_features, convert_dtypes
)
from src.feature_engineering import (
    create_ratio_features, create_interaction_features,
    drop_high_cardinality_categorical, bin_numerical_feature
)
# Import Configurations for inference
from src.config import (
    object_numeric_cols, binary_mapping, missing_numeric_cols, missing_categorical_cols,
    drop_duplicate_subset, datetime_cols, lowercase_cols, outlier_numeric_cols,
    irrelevant_cols, dtype_mappings, tenure_bins, tenure_labels, totalcharges_bins,
    totalcharges_labels, interaction_pairs, ratio_pairs, categorical_cols_list, target_col
)
# Import aws_database for storing predictions
from src.aws_database import store_prediction


MODEL_PATH = os.getenv("MODEL_PATH", "model/best_model.pkl")
TRANSFORMER_PATH = os.getenv("TRANSFORMER_PATH", "model/transformer.pkl")

@st.cache_resource
def load_model_and_transformer():
    # Load model if exists
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    else:
        model = None
        logger.warning("No model found")
    # Load transformer if exists
    if os.path.exists(TRANSFORMER_PATH):
        transformer = joblib.load(TRANSFORMER_PATH)
        logger.info(f"Loaded transformer from {TRANSFORMER_PATH}")
    else:
        transformer = None
        logger.warning("No transformer found; inputs must already be preprocessed similarly to training")
    return model, transformer

model, transformer = load_model_and_transformer()

def full_preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df1 = convert_object_to_numeric(df, columns=object_numeric_cols)
    df2 = encode_binary_columns(df1, mapping=binary_mapping)
    df3 = handle_missing_values(df2, numeric_cols=missing_numeric_cols, categorical_cols=missing_categorical_cols)
    df4 = handle_duplicates(df3, subset=drop_duplicate_subset)
    df5 = standardize_formats(df4, datetime_cols=datetime_cols, lowercase_cols=lowercase_cols)
    df6 = handle_outliers(df5, numeric_cols=outlier_numeric_cols)
    df7 = drop_irrelevant_features(df6, irrelevant_cols)
    df8 = convert_dtypes(df7, dtype_mappings)
    df9 = bin_numerical_feature(df8, column='tenure', bins=tenure_bins, labels=tenure_labels)
    df10 = bin_numerical_feature(df9, column='TotalCharges', bins=totalcharges_bins, labels=totalcharges_labels)
    df11 = create_interaction_features(df10, feature_pairs=interaction_pairs)
    df12 = create_ratio_features(df11, ratio_pairs=ratio_pairs)
    df13, _ = drop_high_cardinality_categorical(df12, categorical_cols=categorical_cols_list)
    if target_col in df13.columns:
        features_df = df13.drop(columns=[target_col])
    else:
        features_df = df13
    return features_df

# Streamlit Web App
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📞",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📞 Telecom Customer Churn Prediction")
st.write(
    "Fill in the customer details below to predict the likelihood of churn. "
    "All features are required for the most accurate prediction."
)

with st.sidebar:
    st.header("About")
    st.write(
        """
        This app predicts the probability of a telecom customer churning using a trained ML model.
        - All preprocessing is handled automatically.
        - Model: Logistic Regression
        - Data: Telco Customer Churn Dataset
        """
    )
    st.write("Developed by Ragiri Himadeep")

def get_feature_inputs():
    customer_id = st.text_input("Customer ID", value=f"customer_{datetime.utcnow().timestamp()}")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0, step=1.0)

    features = {
        "CustomerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return features

user_input = get_feature_inputs()

if st.button("Predict Churn", type="primary"):
    try:
        input_df = pd.DataFrame([user_input])
        features_df = full_preprocess_for_inference(input_df)
        if transformer is not None:
            X = transformer.transform(features_df)
        else:
            X = features_df
        proba = model.predict_proba(X)[0][1]
        pred = int(model.predict(X)[0])
        st.success(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
        st.progress(int(proba * 100))
        st.metric("Churn Probability (%)", f"{proba * 100:.2f}")
        if pred == 1:
            st.warning("⚠️ This customer is likely to churn. Consider retention strategies.")
        else:
            st.info("✅ This customer is unlikely to churn.")
        # Store prediction in DynamoDB
        store_prediction(user_input, pred, proba)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"Prediction failed: {e}")

st.markdown(
    """
    <hr>
    <center>
    <small>
    &copy; 2025 Telecom Customer Churn Prediction App. All rights reserved.
    </small>
    </center>
    """,
    unsafe_allow_html=True
)
