# 📊 Prédiction du Churn des Clients Télécoms

## 📌 Description du Projet
Ce projet vise à prédire si un client d'une entreprise de télécommunications va se désabonner (*churn*) en utilisant des techniques de **Machine Learning** et un **pipeline de données automatisé**.

## 📂 Structure du Projet
```
project/
│   README.md  # Documentation principale
│   requirements.txt  # Dépendances du projet
│   setup.sh  # Script d'installation
│
├── data/
│   ├── customer_churn_telecom_services.csv  # Dataset utilisé
│
├── notebooks/
│   ├── model_comparison.ipynb  # Notebook d'analyse avancée des modèles ML
│
├── src/
│   ├── Pipeline.py  # Pipeline de traitement des données
│   ├── features_engineering.py  # Feature engineering
│   ├── code_projet_final.py  # Modélisation et évaluation des modèles
│   ├── code_visualisation.ipynb  # Analyse exploratoire et visualisation
│
├── deployment/
│   ├── gradient_boosting_model.pkl  # Modèle ML optimisé sauvegardé
│   ├── app.py  # API Flask pour servir le modèle
│   ├── Dockerfile  # Containerisation de l'application
│   ├── config.yml  # Configuration CI/CD
│
└── .gitignore  # Fichier pour exclure certains fichiers du dépôt GitHub
```

## 🚀 Installation et Exécution
### 1️⃣ **Installation des dépendances**
```bash
pip install -r requirements.txt
```

### 2️⃣ **Exécuter le pipeline de données**
```bash
python src/Pipeline.py
```

### 3️⃣ **Entraîner et évaluer le modèle**
```bash
python src/code_projet_final.py
```

### 4️⃣ **Lancer l'API Flask pour les prédictions**
```bash
python deployment/app.py
```

### 5️⃣ **Tester l'API avec une requête POST**
```python
import requests

url = "http://127.0.0.1:5000/predict"
data = [{"tenure": 12, "MonthlyCharges": 75.0, "TotalCharges": 900.0, ...}]  # Compléter avec toutes les features
response = requests.post(url, json=data)
print(response.json())
```

## 🛠️ Technologies Utilisées
- **Python** (pandas, scikit-learn, seaborn, matplotlib, numpy)
- **Flask** (API pour le modèle)
- **Docker** (containerisation de l'application)
- **GitHub Actions** (CI/CD pour l'automatisation du déploiement)

## 📈 Résultats du Modèle
- **Modèle Final** : Gradient Boosting (optimisé avec GridSearchCV)
- **Accuracy** : 81.26%
- **ROC AUC Score** : 86.57%

## 🏆 Objectifs à Long Terme
- Déploiement sur **AWS/GCP** pour une utilisation en production.
- Amélioration du recall pour détecter plus efficacement les churners.


