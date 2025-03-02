#!/bin/bash


pip install -r requirements.txt

echo "Vérification des fichiers du projet..."
[ -f "data/customer_churn_telecom_services.csv" ] || echo "⚠️  Dataset manquant ! Assurez-vous de l'ajouter."
[ -f "deployment/gradient_boosting_model.pkl" ] || echo "⚠️  Modèle non trouvé ! Exécutez le script d'entraînement."

echo "✅ Installation terminée !"
