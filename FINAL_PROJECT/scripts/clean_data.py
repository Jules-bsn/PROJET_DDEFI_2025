import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Définition du chemin d'accès
RAW_DATA_PATH = "data/raw/customer_churn_telecom_services.csv"
PROCESSED_DATA_DIR = "data/processed"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_data.csv")

# Création du répertoire processed si non existant
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Chargement des données
df = pd.read_csv(RAW_DATA_PATH)

# Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()

# Conversion de la colonne cible 'Churn' en valeurs numériques
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Suppression des colonnes non pertinentes identifiées lors de l'analyse
drop_columns = [
    'PhoneService_Yes', 'engagement_score', 'tenure',  # Features avec VIF élevé
    'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
    'StreamingMovies_No internet service', 'StreamingTV_No internet service',
    'TechSupport_No internet service', 'DeviceProtection_No internet service',
    'InternetService_No'  # Colinéarité parfaite
]
df.drop(columns=drop_columns, errors='ignore', inplace=True)

# Feature Engineering : Création de nouvelles features
df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # Évite division par zéro

# Normalisation de certaines variables avec StandardScaler
scaler = StandardScaler()
columns_to_scale = ['TotalCharges', 'avg_monthly_charge']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Remplacement des valeurs manquantes
df.fillna(df.median(), inplace=True)

# Sauvegarde des données nettoyées
df.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"Fichier nettoyé sauvegardé : {PROCESSED_DATA_PATH}")
