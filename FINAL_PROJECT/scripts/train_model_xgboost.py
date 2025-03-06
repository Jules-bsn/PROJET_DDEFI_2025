import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from pipeline import process_pipeline

# Chemins d'accès
RAW_DATA_PATH = 'data/raw/customer_churn_telecom_services.csv'
PROCESSED_DATA_PATH = 'data/processed/cleaned_data.csv'
MODEL_PATH = 'deployment/final_model.pkl'

# Exécution du pipeline de transformation des données
process_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)

# Chargement des données transformées
df = pd.read_csv(PROCESSED_DATA_PATH)
y = df['Churn']
X = df.drop(columns=['Churn'])

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Optimisation des hyperparamètres pour XGBoost
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10]
}
xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'), param_grid_xgb, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
xgb_search.fit(X_train, y_train)

print("Best parameters for XGBoost: ", xgb_search.best_params_)

# Sauvegarde du modèle optimisé
joblib.dump(xgb_search.best_estimator_, MODEL_PATH)
print(f"✅ Modèle sauvegardé : {MODEL_PATH}")
