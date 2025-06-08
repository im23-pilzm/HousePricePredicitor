import pandas as pd
import pickle

test_df = pd.read_csv("../dataset/house-price/processed-data/test_processed.csv")

test_raw = pd.read_csv('../dataset/house-price/raw-data/test.csv')
test_ids = test_raw['Id']

with open("../models/linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(test_df)

output_df = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})

output_df.to_csv("../dataset/house-price/predictions/predictions.csv", index=False)
print("Predictions saved successfully.")

