import pandas as pd
import pickle

# Load raw test data and extract IDs
test_raw = pd.read_csv('../dataset/house-price/raw-data/test.csv')
test_ids = test_raw['Id']

# Drop 'Id' if it's there
if 'Id' in test_raw.columns:
    test_raw.drop(columns=['Id'], inplace=True)

# Load saved preprocessing tools
with open("../models/linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("../models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load expected columns
with open("../models/feature_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)




# Define preprocess function (same as training)
def preprocess(df, imputer, scaler, encoder):
    # Same preprocessing steps as training!
    none_cols = ['Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish',
                 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'PoolQC', 'MiscFeature']

    mode_columns = [
        "MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
        "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
        "Exterior1st", "Exterior2nd", "ExterQual", "ExterCond", "Foundation", "Heating", "HeatingQC",
        "CentralAir", "Electrical", "KitchenQual", "Functional", "PavedDrive", "SaleType", "SaleCondition"
    ]

    df[none_cols] = df[none_cols].fillna("None")
    for col in mode_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df["GarageExists"] = df["GarageType"] != "None"

    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    df[numeric_cols] = imputer.transform(df[numeric_cols])
    cat_encoded = encoder.transform(df[categorical_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    df.drop(columns=categorical_cols, inplace=True, errors='ignore')
    df = pd.concat([df, cat_encoded_df], axis=1)

    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


# Preprocess test data
test_processed = preprocess(test_raw, imputer, scaler, encoder)

# Add any missing columns with 0s, and reorder
for col in expected_columns:
    if col not in test_processed.columns:
        test_processed[col] = 0

# Predict
predictions = model.predict(test_processed)

# Save to CSV
output_df = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})
output_df.to_csv("../dataset/house-price/submission.csv", index=False)

print("Predictions saved to submission.csv")
