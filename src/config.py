object_numeric_cols = ['TotalCharges']

binary_mapping = {
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 0},
    'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'gender': {'Male': 1, 'Female': 0}
}

# After conversion, TotalCharges may have missing values
missing_numeric_cols = ['TotalCharges']
missing_categorical_cols = []

drop_duplicate_subset = []  # No duplicates in Telco

datetime_cols = []  # No datetime columns in Telco
lowercase_cols = []  # No lowercase conversion needed

outlier_numeric_cols = ['MonthlyCharges', 'tenure', 'TotalCharges']

irrelevant_cols = ['customerID']

dtype_mappings = {}  # No extra dtype conversions needed

# Tenure bins (as per EDA)
tenure_bins = [int(0), int(12), int(24), int(48), int(72)]
tenure_labels = ['0-12', '13-24', '25-48', '49+']

# TotalCharges bins (quantile-based)
totalcharges_bins = [0.0, 400.0, 1400.0, 2800.0, 9000.0]
totalcharges_labels = ['0-399', '400-1399', '1400-2799', '2800+']

# Interaction features (as per EDA)
interaction_pairs = [
    ('MonthlyCharges', 'tenure', 'Monthly_x_tenure')
]

# Ratio features (none for Telco, but keep empty for extensibility)
ratio_pairs = []

# Categorical columns to one-hot encode after high-cardinality drop
categorical_cols_list = [
    'InternetService', 'Contract', 'PaymentMethod',
    'tenure_binned', 'TotalCharges_binned'
]

target_col = 'Churn'