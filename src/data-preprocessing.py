from statistics import median

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 200)


def check_missing_values(df):
    if df.empty:
        print("DataFrame is empty.")
        return

    missing_per_col = df.isnull().sum()
    total_missing = missing_per_col.sum()

    if total_missing == 0:
        print("No missing values found.")
        return

    print(f"Found {total_missing} missing values in DF.")

    missing_columns = missing_per_col[missing_per_col > 0]
    print(missing_columns)

    missing_rows = df[df.isnull().any(axis=1)]
    print(missing_rows)


house_df = pd.read_csv("../dataset/house-price/raw-data/test.csv")

none_cols = ['Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish',
             'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
             'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'PoolQC', 'MiscFeature']

house_df[none_cols] = house_df[none_cols].fillna("None")

mode_columns = [
    "MSZoning",
    "Street",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "PavedDrive",
    "SaleType",
    "SaleCondition"
]

for col in mode_columns:
    if house_df[col].isnull().any():
        house_df[col].fillna(house_df[col].mode()[0], inplace=True)


#Fill missing numerical values
numeric_cols = house_df.select_dtypes(include=["number"]).columns
imputer = SimpleImputer(strategy="median")
house_df[numeric_cols] = imputer.fit_transform(house_df[numeric_cols])

#Checks if garage exists
def garage_exists(row):
    return row["GarageType"] != "None"

house_df["GarageExists"] = house_df.apply(garage_exists, axis=1)

#One-hot encoded categorical variables
house_df = pd.get_dummies(house_df)

#Drop unnecessary columns
house_df.drop(columns=["Id"], inplace=True)

#Scale numerical features
scaler = StandardScaler()
house_df[numeric_cols] = scaler.fit_transform(house_df[numeric_cols])

#Check for missing values
check_missing_values(house_df)
