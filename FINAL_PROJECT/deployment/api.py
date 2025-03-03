from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import traceback
import logging
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configuration des logs
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Définir le chemin du modèle
MODEL_PATH = "deployment/final_model.pkl"

# Charger le modèle en toute sécurité
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logging.info("✅ Modèle chargé avec succès !")
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        model = None
else:
    logging.error(f"❌ Modèle introuvable à l'emplacement : {MODEL_PATH}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé ou introuvable"}), 500
    
    try:
        data = request.get_json()
        logging.info(f"🔹 Requête reçue : {data}")

        if not isinstance(data, list):
            return jsonify({"error": "Les données doivent être une liste de dictionnaires"}), 400

        df = pd.DataFrame(data)
        logging.info(f"🔹 Données converties en DataFrame :\n{df.head()}")

        # Prétraitement des données
        df = preprocess_data(df)
        logging.info(f"🔹 Données après preprocessing :\n{df.head()}")

        # Vérifier si les colonnes correspondent à celles du modèle
        model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns
        missing_for_model = [col for col in model_features if col not in df.columns]
        if missing_for_model:
            return jsonify({"error": "Colonnes manquantes après prétraitement", "missing_features": missing_for_model}), 400
        
        df = df[model_features]

        # Prédiction
        prediction = model.predict(df)
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        logging.error(f"❌ Erreur lors de la prédiction : {str(e)}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500

def preprocess_data(df):
    """Applique les transformations nécessaires avant d'envoyer les données au modèle."""
    df = df.copy()
    
    # Feature Engineering
    df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['is_long_term_contract'] = (df['Contract'] == 'Two year').astype(int)
    df['num_services'] = df[['PhoneService', 'MultipleLines', 'InternetService', 
                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                             'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
        lambda row: sum(1 for x in row if x in ['Yes', 'Fiber optic']), axis=1)
    
    # Suppression des colonnes inutiles
    drop_columns = ['CustomerID', 'gender', 'PhoneService', 'tenure', 'MonthlyCharges',
                    'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
                    'StreamingMovies_No internet service', 'StreamingTV_No internet service',
                    'TechSupport_No internet service', 'DeviceProtection_No internet service',
                    'InternetService_No']
    df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore', inplace=True)
    
    # Encodage des variables catégoriques
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    # S'assurer que toutes les colonnes du modèle sont présentes
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else []
    for col in model_features:
        if col not in df.columns:
            df[col] = 0  # Ajouter les colonnes manquantes avec valeur 0
    
    # Réorganiser les colonnes dans le bon ordre
    df = df[model_features]
    
    # Normalisation
    scaler = StandardScaler()
    columns_to_scale = ['TotalCharges', 'avg_monthly_charge', 'num_services']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    # Gestion des valeurs manquantes
    df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)
    
    return df

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
