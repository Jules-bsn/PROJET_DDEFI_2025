import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Charge le dataset depuis le fichier spécifié."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def clean_data(df):
    """Nettoie et transforme les données pour le modèle."""
    df.drop_duplicates(inplace=True)
    
    # Gestion des valeurs manquantes uniquement sur les colonnes numériques
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())
    
    # Conversion de la colonne cible 'Churn' en valeurs numériques
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encodage des variables catégoriques
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Création de nouvelles features
    df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # Évite division par zéro
    df['engagement_score'] = df['tenure'] * 0.2 + df['PaperlessBilling'] * 1.2 + df['Contract'] * 2
    df['is_long_term_contract'] = (df['Contract'] == 2).astype(int)
    
    # Suppression des colonnes avec très faible corrélation avec Churn
    cor_matrix = df.corrwith(df['Churn']).abs()
    low_correlation_features = cor_matrix[cor_matrix < 0.05].index
    df.drop(columns=low_correlation_features, inplace=True)
    
    return df

def remove_multicollinearity(df, threshold=10.0):
    """Supprime les variables fortement colinéaires en utilisant le VIF."""
    df = df.select_dtypes(include=[np.number])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    while vif_data["VIF"].max() > threshold:
        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        df = df.drop(columns=[feature_to_remove])
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return df

def normalize_features(df):
    """Normalise les variables numériques clés."""
    scaler = StandardScaler()
    columns_to_scale = ['TotalCharges', 'avg_monthly_charge', 'engagement_score']
    existing_columns = [col for col in columns_to_scale if col in df.columns]
    if existing_columns:
        df[existing_columns] = scaler.fit_transform(df[existing_columns])
    return df

def balance_classes(X, y):
    """Applique SMOTE pour équilibrer les classes."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def process_pipeline(file_path, output_path):
    """Exécute le pipeline de transformation et sauvegarde les données nettoyées."""
    df = load_data(file_path)
    df = clean_data(df)
    df = remove_multicollinearity(df)
    df = normalize_features(df)
    
    # Séparer les features et la target
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    # Appliquer SMOTE
    X, y = balance_classes(X, y)
    
    # Convertir en DataFrame
    df_resampled = pd.DataFrame(X, columns=X.columns)
    df_resampled['Churn'] = y
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_resampled.to_csv(output_path, index=False)
    print(f"✅ Fichier nettoyé et équilibré sauvegardé : {output_path}")
