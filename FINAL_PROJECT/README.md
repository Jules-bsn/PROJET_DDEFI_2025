# Projet DDEFI 2025 - FINAL_PROJECT

Ce projet met en œuvre un pipeline de machine learning complet pour l'analyse et la prédiction du churn dans un dataset télécom.

## 📂 Structure du projet

```
📂 Projet_Final
├── 📂 data
│   ├── 📂 raw                 # Contient les données d’origine
│   ├── 📂 processed           # Contient les données nettoyées et préparées
│
│
├── 📂 notebook               # Contient les recherches et analyses sur la pipeline et le modéle choisi 
│
├── 📂 scripts                 # Scripts Python pour l'entraînement et l'évaluation
│   ├── data_cleaner.py         # Script pour nettoyer et préparer les données
│   ├── pipeline.py         # regroupe les fonctions utlisées par data_cleaner et api
│   ├── train_model_xgboost.py         # Script pour entraîner le modèle
│   ├── evaluate_model.py      # Script pour évaluer le modèle
│
├── 📂 deployment              # Dossier de déploiement (modèle entraîné et API)
│   ├── final_model.pkl              # Modèle entraîné
│   ├── API.py                 # API pour faire des prédictions
│
├── 📜 requirements.txt         # Liste des dépendances nécessaires
├── 📜 setup.sh                 # Script pour installer l’environnement
├── 📜 README.md                # Explication complète du projet
```

##  Installation

1. **Cloner le projet**
   ```bash
   git clone https://github.com/Jules-bsn/PROJET_DDEFI_2025.git
   cd PROJET_DDEFI_2025/FINAL_PROJECT
   ```

2. **Installer les dépendances**
   ```bash
   bash setup.sh
   ```

##  Utilisation

###  Préparation des données
Avant d'entraîner le modèle, il faut exécuter le script `clean_data.py` pour nettoyer et préparer les données :
```bash
python scripts/data_cleaner.py
```
Cela générera les données nettoyées dans `data/processed/`.

###  Entraînement du modèle
Lancer l'entraînement du modèle :
```bash
python scripts/train_model_xgboost.py
```
Le modèle entraîné sera sauvegardé dans `deployment/final_model.pkl`.

###  Évaluation du modèle
Une fois le modèle entraîné, il est possible de l'évaluer avec :
```bash
python scripts/evaluate_model.py
```

###  Déploiement
Un script `api.py` est fourni pour exposer le modèle via une API Flask.
```bash
python deployment/API.py
```
L'API tournera localement et pourra être utilisée pour faire des prédictions.
Mettre la commande suivante dans un nouveau terminal: 
```bash
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}'

]'
```

## 📜 Notes
- `data_cleaner.py` assure le nettoyage des données et leur transformation aprés les avoir passer dans la pipeline.
- `train_model_xgboost.py` utilise le meilleur modèle identifié avec les hyperparamètres optimaux.
- `evaluate_model.py` génère un rapport d’évaluation basé sur des métriques de classification.
- `API.py` permet d'envoyer des requêtes pour prédire le churn sur de nouvelles données.

---

 **Projet réalisé dans le cadre de DDEFI 2025**
