📚 Rapport Final - Projet DDEFI 2025

1. Introduction

Dans le cadre du cursus DDEFI de Centrale, ce projet vise à concevoir et industrialiser une solution d'analyse de données et de Machine Learning. L'objectif est d'appliquer les concepts étudiés sur les pipelines de données, l'optimisation de modèles et l'industrialisation pour résoudre une problématique business réaliste.

2. Problématique Business

La problématique choisie concerne la prédiction du churn (départ client) dans le secteur des services télécoms. L'objectif est d'identifier les clients susceptibles de quitter l'opérateur et d'anticiper ces départs afin d'améliorer les stratégies de rétention.

3. Pipeline de Données

Le pipeline de données a été conçu pour assurer un traitement automatisé des données.

✅ Prétraitement des Données

Chargement et nettoyage du dataset.

Encodage des variables catégoriques.

Gestion des valeurs manquantes.

Normalisation des variables numériques (‘TotalCharges’, ‘MonthlyCharges’, ‘engagement_score’).

Suppression des variables redondantes (‘num_services’, ‘PhoneService_Yes’).

Gestion de la collinéarité via Variance Inflation Factor (VIF).

4. Modélisation & Optimisation

🎯 Modèles testés :

Logistic Regression

Random Forest

Gradient Boosting (GBM)

XGBoost ✅ (Meilleur modèle retenu)

SVM

KNN

🔄 Optimisation des Hyperparamètres :

Tuning de ‘max_depth’ et ‘learning_rate’ pour XGBoost et Gradient Boosting via RandomizedSearchCV.

Validation croisee stratifiée (« Stratified KFold »).

Gestion du déséquilibre des classes avec SMOTE.

🎯 Résultats obtenus :

Meilleur modèle : XGBoost avec ROC-AUC = 0.9114.

SHAP analysis a mis en évidence les variables les plus importantes :

InternetService_Fiber optic

PaymentMethod_Electronic check

avg_monthly_charge

TotalCharges

Contract_Two year

5. Explicabilité des Modèles

Nous avons utilisé SHAP (Shapley Additive Explanations) pour analyser l'impact des variables sur la prédiction du churn.

Les clients avec un abonnement "Fiber Optic" sont plus susceptibles de partir.

Le paiement par "Electronic Check" est fortement corrélé avec le churn.

Les contrats long-terme réduisent le risque de churn.

6. Industrialisation

Stockage et déploiement du code : GitHub & conteneurisation Docker.

Possibilité de déploiement sous forme d'API (Flask/FastAPI).

Monitoring & Améliorations : Intégration continue et mise à jour des modèles sur de nouvelles données.

7. Conclusion & Perspectives

Ce projet a permis de construire un pipeline end-to-end allant de la préparation des données à l'optimisation du modèle et son explicabilité. Les prochaines étapes incluent le déploiement sous forme d'API et l'optimisation des performances du modèle avec de nouvelles données.

📈 Mots-clés : Machine Learning, XGBoost, Churn Prediction, Pipelines de Données, Industrialisation.

