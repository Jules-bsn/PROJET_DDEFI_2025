import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import joblib

def train_xgboost(input_path, model_output_path):
    """
    Charge les données nettoyées, entraîne un modèle XGBoost avec optimisation des hyperparamètres
    et enregistre le modèle entraîné.
    """
    print("\n🔹 Chargement des données nettoyées...")
    df = pd.read_csv(input_path)
    
    print("🔹 Séparation des features et de la cible...")
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    print("🔹 Séparation en jeu d'entraînement et de validation...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🔹 Définition des hyperparamètres pour la recherche...")
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'n_estimators': [100, 200, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5,
                                scoring='roc_auc', random_state=42, verbose=1, n_jobs=-1)
    
    print("🔹 Entraînement du modèle XGBoost avec recherche d'hyperparamètres...")
    search.fit(X_train, y_train)
    
    print(f"✅ Meilleurs hyperparamètres trouvés : {search.best_params_}")
    best_model = search.best_estimator_
    
    print("🔹 Évaluation du modèle...")
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"✅ Score ROC-AUC sur le jeu de validation : {auc_score:.4f}")
    
    print("🔹 Sauvegarde du modèle entraîné...")
    joblib.dump(best_model, model_output_path)
    print(f"✅ Modèle enregistré sous : {model_output_path}")

if __name__ == "__main__":
    train_xgboost("data/processed/cleaned_data.csv", "deployment/final_model.pkl")
