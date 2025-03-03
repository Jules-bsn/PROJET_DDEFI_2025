import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# 📚 Chargement des données pré-traitées
df = pd.read_csv('data/processed/cleaned_data.csv')
y = df['Churn']
X = df.drop(columns=['Churn'])

# 📌 Application de SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 📌 Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# 🎯 Optimisation des hyperparamètres XGBoost
best_xgb_params = {'max_depth': 10, 'learning_rate': 0.2}
best_xgb = XGBClassifier(**best_xgb_params, eval_metric='logloss', use_label_encoder=False)
best_xgb.fit(X_train, y_train)

# 📁 Sauvegarde du modèle optimisé
os.makedirs('deployment', exist_ok=True)
joblib.dump(best_xgb, 'deployment/final_model.pkl')

# 📌 Évaluation avec validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_xgb = RandomizedSearchCV(best_xgb, param_distributions={}, n_iter=1, cv=cv, scoring='roc_auc')
scores_xgb.fit(X_resampled, y_resampled)
print(f"✅ XGBoost Mean ROC-AUC: {scores_xgb.best_score_:.4f}")

print("📂 Modèle XGBoost optimisé et sauvegardé !")
