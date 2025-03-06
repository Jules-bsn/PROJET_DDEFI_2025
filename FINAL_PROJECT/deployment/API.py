from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os
from scripts/pipeline import clean_data, normalize_features, remove_multicollinearity

# Configuration des logs
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# 📁 Définition des chemins
MODEL_PATH = "deployment/final_model.pkl"

# Charger le modèle
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(" Modèle chargé avec succès !")
    except Exception as e:
        logging.error(f" Erreur lors du chargement du modèle : {str(e)}")
        model = None
else:
    logging.error(f" Modèle introuvable à l'emplacement : {MODEL_PATH}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    """Route pour effectuer des prédictions avec le modèle."""
    if model is None:
        return jsonify({"error": "Modèle non chargé ou introuvable"}), 500
    
    try:
        data = request.get_json()
        logging.info(f" Requête reçue : {data}")
        
        if not isinstance(data, list):
            return jsonify({"error": "Les données doivent être une liste de dictionnaires"}), 400
        
        df = pd.DataFrame(data)
        logging.info(f" Données converties en DataFrame :\n{df.head()}")
        
        # Prétraitement des données
        df = clean_data(df)
        df = remove_multicollinearity(df)
        df = normalize_features(df)
        logging.info(f" Données après preprocessing :\n{df.head()}")
        
        # Vérifier que les colonnes correspondent à celles du modèle
        model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns
        missing_for_model = [col for col in model_features if col not in df.columns]
        if missing_for_model:
            return jsonify({"error": "Colonnes manquantes après prétraitement", "missing_features": missing_for_model}), 400
        
        df = df[model_features]
        
        # Prédiction
        prediction = model.predict(df)
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        logging.error(f" Erreur lors de la prédiction : {str(e)}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
