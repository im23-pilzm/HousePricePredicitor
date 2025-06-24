# House Price Predictor

This project is a machine learning pipeline designed to predict house prices based on various features. It includes data preprocessing, model training, and testing, as well as utilities for evaluating predictions.

## Project Structure
Later

## Features

- **Data Preprocessing**:
  - Handles missing values.
  - Encodes categorical variables.
  - Scales numeric features.
  - Ensures consistency between training and test datasets.

- **Model Training**:
  - Trains a linear regression model using preprocessed data.
  - Saves trained models and feature columns for future use.

- **Testing**:
  - Validates the model's predictions.
  - Ensures the shape and consistency of predictions.

## Requirements

- Python 3.11
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `pickle`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/im23-pilzm/HousePricePredictor.git
   cd HousePricePredictor

2. Install dependencies:
pip install -r requirements.txt

## Usage
Data Preprocessing
Run the preprocessing script to clean and prepare the data:
python src/data-preprocessing.py

Model Training
Train the model using the preprocessed data:
python src/model-training.py

Testing
Run the test suite to validate the model:
python test/test_data.py
python test/test_model.py






   
