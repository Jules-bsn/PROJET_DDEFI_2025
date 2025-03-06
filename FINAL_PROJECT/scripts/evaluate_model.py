import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def evaluate_model(model_path, data_path):
    """Charge le modèle et les données, effectue l'évaluation et détecte l'overfitting."""
    print("🔹 Chargement du modèle...")
    model = joblib.load(model_path)
    
    print("🔹 Chargement des données...")
    df = pd.read_csv(data_path)
    
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    print("🔹 Séparation des données en train et test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🔹 Prédictions du modèle...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    print("🔹 Calcul des métriques...")
    metrics = {
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_pred),
        "Test Precision": precision_score(y_test, y_test_pred),
        "Train Recall": recall_score(y_train, y_train_pred),
        "Test Recall": recall_score(y_test, y_test_pred),
        "Train F1-score": f1_score(y_train, y_train_pred),
        "Test F1-score": f1_score(y_test, y_test_pred),
        "Train ROC-AUC": roc_auc_score(y_train, y_train_proba),
        "Test ROC-AUC": roc_auc_score(y_test, y_test_proba)
    }
    
    print("🔹 Détection d'overfitting...")
    overfitting_status = "Yes" if metrics["Train Accuracy"] - metrics["Test Accuracy"] > 0.1 else "No"
    metrics["Overfitting Detected"] = overfitting_status
    
    print("🔹 Affichage des résultats...")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nConfusion Matrix (Test Set):\n", confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report (Test Set):\n", classification_report(y_test, y_test_pred))
    
    return metrics

if __name__ == "__main__":
    model_file = "deployment/final_model.pkl"
    data_file = "data/processed/cleaned_data.csv"
    evaluate_model(model_file, data_file)
