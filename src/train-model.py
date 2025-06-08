import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('../dataset/house-price/processed-data/train_processed.csv')

x = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_val)

mae = mean_absolute_error(y_val, y_pred)
print(f"Validation MAE: {mae:.2f}")

with open('../models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved.")
