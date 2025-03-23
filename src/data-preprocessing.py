import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 200)

df = pd.read_csv("../dataset/house-price/raw-data/test.csv")

if df.empty:
    print("DataFrame is empty!")
else:












#print(df.info())

#drop rows with missing data
#df.isnull().sum()
#df.dropna(inplace=True)

#remove duplicates
#df.duplicated().sum()
#df.drop_duplicates(inplace=True)

# List of columns to scale
columns_to_scale = [
    "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
    "2ndFlrSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal"
]

#standardize & normalize data
#st_scaler = MinMaxScaler()
#df[columns_to_scale] = st_scaler.fit_transform(df[columns_to_scale])
#no_scaler = StandardScaler()
#df[columns_to_scale] = no_scaler.fit_transform(df[columns_to_scale])
#df.dropna(inplace=True)
#if df.empty:
    #print("Error: DataFrame is empty after dropping NaN values.")
    #exit()  # Stop execution to avoid further errors

#df.info()
#df.describe()
