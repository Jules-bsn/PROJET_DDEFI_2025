import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_xgboost(data_file, model_output):
    """Entraîne un modèle XGBoost et sauvegarde le modèle entraîné."""
    print("🔹 Chargement des données nettoyées...")
    df = pd.read_csv(data_file)
    
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    print("🔹 Séparation des données en train et test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🔹 Entraînement du modèle XGBoost...")
    model = xgb.XGBClassifier(
        eval_metric='logloss',
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("🔹 Évaluation du modèle...")
    y_pred = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"✅ Score ROC-AUC: {roc_auc:.4f}")
    
    print("🔹 Sauvegarde du modèle...")
    joblib.dump(model, model_output)
    print(f"✅ Modèle sauvegardé sous {model_output}")

if __name__ == "__main__":
    input_data = "data/processed/cleaned_data.csv"
    model_path = "deployment/final_model.pkl"
    train_xgboost(input_data, model_path)
