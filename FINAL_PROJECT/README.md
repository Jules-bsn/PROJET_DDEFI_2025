rédiction du Churn des Clients Télécom

📂 Structure du Projet

📂 customer_churn_prediction
│── 📂 data/                 # Contient les datasets bruts et transformés
│   ├── customer_churn_telecom_services.csv
│── 📂 notebooks/            # Notebooks pour l'analyse exploratoire et la modélisation
│   ├── 1_data_exploration.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_selection.ipynb
│── 📂 scripts/              # Scripts Python pour l'industrialisation
│   ├── pipeline.py          # Pipeline complet de traitement des données
│   ├── train_model.py       # Entraînement du modèle retenu
│   ├── predict.py           # Script pour faire des prédictions
│── 📂 deployment/           # Contient le modèle sauvegardé
│   ├── model.pkl
│── requirements.txt         # Dépendances du projet
│── README.md                # Documentation du projet
│── .gitignore               # Fichiers à ignorer dans le dépôt Git

-Objectif du Projet

Prédire si un client va quitter l’opérateur télécom (churn) en utilisant un modèle de Machine Learning.

-Étapes du Projet

Exploration des Données (``)

Analyse statistique et visualisations

Identification des variables importantes

Feature Engineering (``)

Création de nouvelles variables

Normalisation des données

-Sélection du Modèle (``)

Test de plusieurs algorithmes (Random Forest, SVM, etc.)

Choix du meilleur modèle (Random Forest)

-Industrialisation (``)

pipeline.py : Prétraitement et transformation des données

train_model.py : Entraînement et sauvegarde du modèle

predict.py : Chargement du modèle et prédictions

-Utilisation

-Installation des dépendances

pip install -r requirements.txt

-Entraîner le modèle

python scripts/train_model.py

 -Faire une prédiction

python scripts/predict.py

-Résultats du Modèle

Modèle

Accuracy

AUC

Random Forest

0.85

0.91

Gradient Boosting

0.83

0.89

Logistic Regression

0.80

0.86

SVM

0.78

0.82

Le Random Forest a été retenu pour ses meilleures performances sur les données.

Contact

Pour toute question ou amélioration du projet, n’hésitez pas à contribuer sur GitHub ! 

