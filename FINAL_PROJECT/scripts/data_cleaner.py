import pandas as pd
from pipeline import preprocess_data, apply_smote, remove_high_vif_features

def clean_data(input_path, output_path):
    """
    Charge les données, applique le prétraitement et enregistre les données nettoyées.
    """
    print("Chargement des données...")
    df = pd.read_csv(input_path)
    
    print("Prétraitement des données...")
    df = preprocess_data(df)
    
    print("Séparation des features et de la cible...")
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    print("Application de SMOTE pour équilibrer les classes...")
    X_resampled, y_resampled = apply_smote(X, y)
    
    print("Suppression des variables à forte colinéarité...")
    X_resampled = remove_high_vif_features(X_resampled)
    
    print("Sauvegarde des données nettoyées...")
    cleaned_data = pd.concat([X_resampled, y_resampled], axis=1)
    cleaned_data.to_csv(output_path, index=False)
    
    print("Nettoyage terminé. Fichier sauvegardé à:", output_path)

if __name__ == "__main__":
    clean_data("data/raw/customer_churn_telecom_services.csv", "data/processed/cleaned_data.csv")
