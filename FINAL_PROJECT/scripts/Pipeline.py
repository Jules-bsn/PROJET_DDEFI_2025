"""
PROJET FINAL - Prédiction churns opérateurs téléphoniques
Dataset Churns: https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services?resource=download
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer

# Load dataset
print(os.getcwd())
os.chdir('/Users/julesbesson/Documents/CENTRALE marseille/S9/PROJET_DDEFI_2025/FINAL_PROJECT')
dataframe_brut = pd.read_csv('customer_churn_telecom_services.csv')

def pre_traitement(dataframe):
    """
    - Convert 'TotalCharges' to numeric
    - Handle missing values
    - Encode categorical features
    """
    df = dataframe.copy()

    required_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    # Ensure all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert 'TotalCharges' to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df

# Missing values check function
def verifier_valeurs_manquantes(df):
    print("Checking for missing values:")
    missing = df.isna().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found.")

# Encoding function
def encoder(dataframe):
    df = dataframe.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# Pipeline function
def pipeline(dataframe):
    df = dataframe.copy()
    df = pre_traitement(df)
    verifier_valeurs_manquantes(df)
    df = encoder(df)

    # Splitting data
    X = df.drop(columns=['Churn_Yes'])  # Ensure target column is correctly named
    y = df['Churn_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/customer_churn_telecom_services.csv")
    X_train, X_test, y_train, y_test = pipeline(df)
    print(f"Processed data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")
