import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 200)

df = pd.read_csv("../dataset/house-price/test.csv")

print(df.columns.tolist())

print(df.dtypes.T)
