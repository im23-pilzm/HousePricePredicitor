import pandas as pd
from sklearn.preprocessing import StandardScaler
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

def preprocess(df, imputer=None, scaler=None, is_train=True):
    df[none_cols] = df[none_cols].fillna("None")

    for col in mode_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df["GarageExists"] = df.apply(garage_exists, axis=1)

    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns

    if is_train:
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = imputer.transform(df[numeric_cols])

    df = pd.get_dummies(df)

    if is_train:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, imputer, scaler

# Load datasets
train_raw = pd.read_csv("../dataset/house-price/raw-data/train.csv")
test_raw = pd.read_csv("../dataset/house-price/raw-data/test.csv")

# Separate SalePrice (target) from train
y_train = train_raw["SalePrice"]
train_raw.drop(columns=["SalePrice"], inplace=True)

# Preprocess both
train_processed, imputer, scaler = preprocess(train_raw, is_train=True)
test_processed, _, _ = preprocess(test_raw, imputer=imputer, scaler=scaler, is_train=False)

# Add target column back to train
train_processed["SalePrice"] = y_train

# Save
train_processed.to_csv("../dataset/house-price/processed-data/train_processed.csv", index=False)
test_processed.to_csv("../dataset/house-price/processed-data/test_processed.csv", index=False)

print("Train and test sets preprocessed and saved.")