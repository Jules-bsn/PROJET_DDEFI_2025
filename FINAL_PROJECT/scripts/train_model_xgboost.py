import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
import joblib
import os

df = pd.read_csv('data/processed/cleaned_data.csv')
y = df['Churn']
X = df.drop(columns=['Churn'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10]
}
xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                                 param_grid_xgb, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
xgb_search.fit(X_train, y_train)

print("Best parameters for XGBoost: ", xgb_search.best_params_)

# Sauvegarde du modèle optimisé
os.makedirs('deployment', exist_ok=True)
joblib.dump(xgb_search.best_estimator_, 'deployment/final_model.pkl')

# validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_xgb = RandomizedSearchCV(xgb_search.best_estimator_, param_distributions={}, n_iter=1, cv=cv, scoring='roc_auc')
scores_xgb.fit(X, y)
print(f" XGBoost Mean ROC-AUC: {scores_xgb.best_score_:.4f}")

print(" Modèle XGBoost optimisé et sauvegardé !")
