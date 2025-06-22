import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 200)

# Columns that get "None" if missing
none_cols = ['Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish',
             'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
             'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'PoolQC', 'MiscFeature']

# Categorical columns filled by mode
mode_columns = [
    "MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "ExterQual", "ExterCond", "Foundation", "Heating", "HeatingQC",
    "CentralAir", "Electrical", "KitchenQual", "Functional", "PavedDrive", "SaleType", "SaleCondition"
]


def garage_exists(row):
    return row["GarageType"] != "None"


def preprocess(df, imputer=None, scaler=None, encoder=None, is_train=True):
    df = df.copy()

    df[none_cols] = df[none_cols].fillna("None")

    for col in mode_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df["GarageExists"] = df.apply(garage_exists, axis=1)

    # Save Id for later if needed
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Imputer
    if is_train:
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = imputer.transform(df[numeric_cols])

    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if is_train:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoded = encoder.fit_transform(df[categorical_cols])
    else:
        cat_encoded = encoder.transform(df[categorical_cols])

    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols),
                                  index=df.index)

    # Drop original categorical columns
    df.drop(columns=categorical_cols, inplace=True, errors='ignore')

    # Combine numeric + encoded categorical
    df = pd.concat([df, cat_encoded_df], axis=1)

    # Scaler
    if is_train:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, imputer, scaler, encoder


# Load datasets
train_raw = pd.read_csv("../dataset/house-price/raw-data/train.csv")
test_raw = pd.read_csv("../dataset/house-price/raw-data/test.csv")

# Separate SalePrice (target) from train
y_train = train_raw["SalePrice"]
train_raw.drop(columns=["SalePrice"], inplace=True)

# Preprocess both
train_processed, imputer, scaler, encoder = preprocess(train_raw, is_train=True)
test_processed, _, _, _ = preprocess(test_raw, imputer=imputer, scaler=scaler, encoder=encoder, is_train=False)

# Add target column back to train
train_processed["SalePrice"] = y_train

# Save processed datasets
train_processed.to_csv("../dataset/house-price/processed-data/train_processed.csv", index=False)
test_processed.to_csv("../dataset/house-price/processed-data/test_processed.csv", index=False)

# Save the imputer, scaler, encoder for later use
with open("../models/imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("../models/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Save columns for later use in prediction
with open("../models/feature_columns.pkl", "wb") as f:
    pickle.dump(train_processed.drop(columns=["SalePrice"]).columns.tolist(), f)

print("Train and test sets preprocessed and saved.")
