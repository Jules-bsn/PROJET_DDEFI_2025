import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 📌 Définition des chemins d'accès
RAW_DATA_PATH = "data/raw/customer_churn_telecom_services.csv"
PROCESSED_DATA_DIR = "data/processed"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_data.csv")

# 📌 Création du répertoire processed si non existant
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 📌 Chargement des données
df = pd.read_csv(RAW_DATA_PATH)

# 📌 Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()

# 📌 Conversion de la colonne cible 'Churn' en valeurs numériques
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 📌 Feature Engineering : Création de nouvelles variables
df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # Évite division par zéro
df['is_long_term_contract'] = (df['Contract'] == 'Two year').astype(int)
df['num_services'] = df[['PhoneService', 'MultipleLines', 'InternetService', 
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                         'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
    lambda row: sum(1 for x in row if x in ['Yes', 'Fiber optic']), axis=1
)

# 📌 Suppression des colonnes non pertinentes identifiées lors de l'analyse
drop_columns = [
    'PhoneService', 'engagement_score', 'tenure', 'MonthlyCharges',  # Suppression après feature engineering
    'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
    'StreamingMovies_No internet service', 'StreamingTV_No internet service',
    'TechSupport_No internet service', 'DeviceProtection_No internet service',
    'InternetService_No'  # Colinéarité parfaite
]
df.drop(columns=drop_columns, errors='ignore', inplace=True)

# 📌 Normalisation des variables importantes
scaler = StandardScaler()
columns_to_scale = ['TotalCharges', 'avg_monthly_charge', 'num_services']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# 📌 Remplacement des valeurs manquantes
df.fillna(df.median(), inplace=True)

# 📌 Sauvegarde des données nettoyées
df.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"✅ Fichier nettoyé sauvegardé : {PROCESSED_DATA_PATH}")
