import pandas as pd
import pytest
from src.data_preprocessing import preprocess


# Sample data fixture
@pytest.fixture
def sample_data():
    return pd.read_csv("../test/fixtures/sample_raw.csv")


def test_preprocess_output_shape(sample_data):
    processed_df, imputer, scaler, encoder = preprocess(sample_data, is_train=True)

    assert not processed_df.isnull().any().any(), "There are missing values in the processed data"
    assert "GarageExists" in processed_df.columns, "'GarageExists' column not found after preprocessing"
    assert processed_df.shape[0] == sample_data.shape[0], "Number of rows changed during preprocessing"


def test_transform_consistency(sample_data):
    # Train phase
    train_df, imputer, scaler, encoder = preprocess(sample_data, is_train=True)
    # Inference phase
    test_df, _, _, _ = preprocess(sample_data, imputer=imputer, scaler=scaler, encoder=encoder, is_train=False)

    assert all(train_df.columns == test_df.columns), "Train and test column mismatch after transformation"
