import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier
import joblib

# Chargement des données pré-traitées
df = pd.read_csv('../data/processed/cleaned_data.csv')
y = df['Churn']
X = df.drop(columns=['Churn'])

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimisation des hyperparamètres
param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 10]}
xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'), param_grid, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
xgb_search.fit(X_train, y_train)

# Sauvegarde du modèle optimisé
joblib.dump(xgb_search.best_estimator_, '../deployment/final_model.pkl')
print("Modèle sauvegardé avec succès!")
