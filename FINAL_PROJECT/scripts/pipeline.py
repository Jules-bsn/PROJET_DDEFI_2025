import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """Charge le dataset depuis le fichier spécifié."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def clean_data(df):
    """Nettoie et transforme les données pour le modèle."""
    # Suppression des doublons
    df.drop_duplicates(inplace=True)
    
    # Gestion des valeurs manquantes uniquement sur les colonnes numériques
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())
    
    # Conversion de la colonne cible 'Churn' en valeurs numériques
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Création de nouvelles features
    df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # Évite division par zéro
    df['is_long_term_contract'] = (df['Contract'] == 'Two year').astype(int)
    df['num_services'] = df[['PhoneService', 'MultipleLines', 'InternetService', 
                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                             'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
        lambda row: sum(1 for x in row if x in ['Yes', 'Fiber optic']), axis=1
    )
    
    # Suppression des colonnes non pertinentes
    drop_columns = [
        'CustomerID', 'gender', 'PhoneService', 'tenure', 'MonthlyCharges',
        'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
        'StreamingMovies_No internet service', 'StreamingTV_No internet service',
        'TechSupport_No internet service', 'DeviceProtection_No internet service',
        'InternetService_No'
    ]
    df.drop(columns=drop_columns, errors='ignore', inplace=True)
    
    # Encodage des variables catégoriques
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    return df

def normalize_features(df):
    """Normalise les variables numériques clés."""
    scaler = StandardScaler()
    columns_to_scale = ['TotalCharges', 'avg_monthly_charge', 'num_services']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df

def process_pipeline(file_path, output_path):
    """Exécute le pipeline de transformation et sauvegarde les données nettoyées."""
    df = load_data(file_path)
    df = clean_data(df)
    df = normalize_features(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Fichier nettoyé sauvegardé : {output_path}")
