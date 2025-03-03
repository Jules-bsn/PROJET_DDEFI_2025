# Détection du Churn - Projet DDEFI 2025

Ce projet vise à prédire le churn des clients en utilisant des techniques de Machine Learning. Il comprend des étapes de prétraitement des données, de sélection de features, d'entraînement et d'optimisation de modèle.

## 📁 Structure du projet

```
├── data/
│   ├── raw/               # Données brutes
│   ├── processed/         # Données prétraitées
├── scripts/
│   ├── clean_data.py      # Script de nettoyage et prétraitement des données
│   ├── train_model.py     # Entraînement et évaluation du modèle
│   ├── inference.py       # Prédiction sur de nouvelles données
├── pipeline/
│   ├── feature_engineering.py  # Transformation des features
│   ├── model_training.py       # Entraînement des modèles optimisés
│   ├── evaluation.py           # Évaluation des modèles
├── deployment/
│   ├── app.py              # API pour le modèle
│   ├── Dockerfile          # Fichier Docker pour déploiement
│   ├── requirements.txt    # Dépendances
│   ├── setup.sh            # Script d'installation
├── notebooks/
│   ├── exploration.ipynb   # Analyse exploratoire des données
│   ├── model_training.ipynb # Expérimentation sur le modèle
│   ├── evaluation.ipynb    # Évaluation des performances
├── README.md               # Guide du projet
```

## 🚀 Installation et Configuration

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/ton-repo.git
   cd ton-repo
   ```

2. **Créer un environnement virtuel :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # sous Mac/Linux
   venv\Scripts\activate   # sous Windows
   ```

3. **Installer les dépendances :**
   ```bash
   pip install -r deployment/requirements.txt
   ```

4. **Exécuter le script de nettoyage des données :**
   ```bash
   python scripts/clean_data.py
   ```

5. **Lancer l'entraînement du modèle :**
   ```bash
   python scripts/train_model.py
   ```

6. **Effectuer des prédictions :**
   ```bash
   python scripts/inference.py --input_file data/processed/test_data.csv
   ```

## 📊 Résumé du Modèle

- **Modèle sélectionné** : XGBoost optimisé
- **Meilleurs hyperparamètres** :
  - `learning_rate = 0.2`
  - `max_depth = 10`
- **Performance** :
  - **ROC-AUC** = 0.92

## 🚀 Déploiement

1. **Construire l’image Docker :**
   ```bash
   docker build -t churn-prediction .
   ```
2. **Lancer le conteneur :**
   ```bash
   docker run -p 5000:5000 churn-prediction
   ```
3. **Tester l'API :**
   ```bash
   curl -X POST http://localhost:5000/predict -d '{"feature1": value1, "feature2": value2}'
   ```

## 📌 Contributeurs

- **[Ton Nom]** - Data Scientist
- **[Collaborateurs]** - Développeurs & Analysts

---

Ce projet est un travail en cours. N'hésite pas à proposer des améliorations via des issues ou des PRs. 🚀
