import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from pipeline import process_pipeline
import os

# 📁 Définition des chemins
MODEL_PATH = "deployment/final_model.pkl"
RAW_DATA_PATH = "data/raw/customer_churn_telecom_services.csv"
CLEAN_DATA_PATH = "data/processed/cleaned_data.csv"

def load_model():
    """Charge le modèle XGBoost depuis le fichier pickle."""
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Modèle chargé avec succès !\n")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        exit(1)

def load_data():
    """Vérifie et charge les données nettoyées, sinon les génère avec process_pipeline."""
    if not os.path.exists(CLEAN_DATA_PATH):
        print("🔄 Fichier nettoyé introuvable, exécution du pipeline de nettoyage...")
        process_pipeline(RAW_DATA_PATH, CLEAN_DATA_PATH)
    
    data = pd.read_csv(CLEAN_DATA_PATH)
    print(f"✅ Données nettoyées chargées ({data.shape[0]} échantillons) !\n")
    
    X = data.drop(columns=['Churn'])  # Supposons que 'Churn' est la cible
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"✅ Données divisées : {X_test.shape[0]} échantillons pour le test\n")
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle sur l'ensemble de test."""
    print("\n================ ÉVALUATION DU MODÈLE ================\n")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba)
    }
    
    # Affichage formaté
    for metric, value in metrics.items():
        print(f"📊 {metric: <12}: {value:.4f}")
    
    # Affichage de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de Confusion :")
    print(cm, "\n")
    
    # Affichage du rapport de classification
    print("Rapport de Classification :")
    print(classification_report(y_test, y_pred))
    
    print("\n====================================================\n")

def main():
    """Exécute l'évaluation du modèle."""
    model = load_model()
    X_test, y_test = load_data()
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
