import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb

# Chargement du modèle
MODEL_PATH = "deployment/model.pkl"
CLEAN_DATA_PATH = "data/processed/cleaned_data.csv"  # Données nettoyées générées par clean_data.py

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Modèle chargé avec succès !\n")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        exit(1)

def load_data():
    try:
        data = pd.read_csv(CLEAN_DATA_PATH)
        print(f"✅ Données nettoyées chargées ({data.shape[0]} échantillons) !\n")
    except FileNotFoundError:
        print("❌ Erreur : Impossible de charger les données nettoyées. Exécutez clean_data.py d'abord.")
        exit(1)
    
    X = data.drop(columns=['Churn'])  # Supposons que 'Churn' est la cible
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Données divisées : {X_test.shape[0]} échantillons pour le test\n")
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    print("\n================ ÉVALUATION DU MODÈLE ================\n")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Affichage formaté
    print(f"📊 Accuracy        : {accuracy:.4f}")
    print(f"📊 Precision       : {precision:.4f}")
    print(f"📊 Recall          : {recall:.4f}")
    print(f"📊 F1-Score        : {f1:.4f}")
    print(f"📊 AUC-ROC         : {roc_auc:.4f}\n")
    
    # Affichage de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de Confusion :")
    print(cm, "\n")
    
    # Affichage du rapport de classification
    print("Rapport de Classification :")
    print(classification_report(y_test, y_pred))
    
    print("\n====================================================\n")

def main():
    model = load_model()
    X_test, y_test = load_data()
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
