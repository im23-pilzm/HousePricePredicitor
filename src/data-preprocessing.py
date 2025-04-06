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

#TODO create a function that checks if a garage exists


check_missing_values(house_df)
