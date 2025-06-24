import pickle
import numpy as np
import pandas as pd
import os


def test_model_prediction_shape():
    # Load test data
    test_data_path = ("../dataset/house-price/processed-data/test_processed.csv")
    test_df = pd.read_csv(test_data_path)

    # Drop ID column if it exists
    if "Id" in test_df.columns:
        test_df = test_df.drop(columns=["Id"])

    # Load feature columns used during training
    with open("../models/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Ensure test_df has the same features (fill missing columns with 0 if necessary)
    for col in feature_columns:
        if col not in test_df.columns:
            test_df[col] = 0
    test_df = test_df[feature_columns]

    # Load model
    with open("../models/linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Make predictions
    predictions = model.predict(test_df)

    # Assertions
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == test_df.shape[0]

