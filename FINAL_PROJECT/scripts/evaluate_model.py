import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_data_path):
    """
    Charge le modèle entraîné, évalue ses performances et détecte le sur-apprentissage.
    """
    print("\n🔹 Chargement du modèle XGBoost...")
    model = joblib.load(model_path)
    print("✅ Modèle chargé avec succès.")
    
    print("🔹 Chargement des données de test...")
    df = pd.read_csv(test_data_path)
    y_test = df['Churn']
    X_test = df.drop(columns=['Churn'])
    
    print("🔹 Prédictions du modèle...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("🔹 Calcul des métriques de performance...")
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"✅ ROC-AUC Score : {auc_score:.4f}")
    print("\nClassification Report :\n", classification_report(y_test, y_pred))
    
    print("🔹 Matrice de confusion...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion")
    plt.show()
    
    print("🔹 Vérification du sur-apprentissage...")
    train_auc = model.best_score if hasattr(model, 'best_score') else None
    if train_auc:
        print(f"AUC Score sur l'entraînement : {train_auc:.4f}")
        print(f"AUC Score sur le test : {auc_score:.4f}")
        if train_auc - auc_score > 0.05:
            print("⚠️ Attention : Le modèle pourrait être en sur-apprentissage.")
    else:
        print("⚠️ Impossible de vérifier le sur-apprentissage (aucune métrique enregistrée sur le training).")
    
if __name__ == "__main__":
    evaluate_model("deployment/final_model.pkl", "data/processed/cleaned_data.csv")
