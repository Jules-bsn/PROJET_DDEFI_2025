import os
import pandas as pd
from pipeline import load_data, preprocess_data, balance_data, remove_high_vif_features

def clean_data(input_file, output_file):
    """Charge, nettoie et enregistre les données transformées."""
    print("🔹 Chargement des données...")
    df = load_data(input_file)
    
    print("🔹 Prétraitement des données...")
    df = preprocess_data(df)
    
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    print("🔹 Équilibrage des classes avec SMOTE...")
    X_resampled, y_resampled = balance_data(X, y)
    
    print("🔹 Suppression des variables avec colinéarité excessive...")
    X_resampled = remove_high_vif_features(X_resampled)
    
    df_cleaned = X_resampled.copy()
    df_cleaned['Churn'] = y_resampled
    
    # Vérifier et créer le dossier de sortie si nécessaire
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"✅ Sauvegarde des données nettoyées dans {output_file}")
    df_cleaned.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_path = "data/raw/customer_churn_telecom_services.csv"
    output_path = "data/processed/cleaned_data.csv"
    clean_data(input_path, output_path)
