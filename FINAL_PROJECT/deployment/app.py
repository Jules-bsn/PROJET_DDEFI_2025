from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Charger le modèle
import os
model_path = os.path.join(os.path.dirname(__file__), "gradient_boosting_model.pkl")
model = joblib.load(model_path)


# Initialiser l'API Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données en JSON
        data = request.get_json()
        
        # Convertir en DataFrame
        df = pd.DataFrame(data)
        
        # Vérifier que toutes les features sont présentes
        expected_features = model.feature_names_in_
        if not all(feature in df.columns for feature in expected_features):
            return jsonify({"error": "Données d'entrée incomplètes. Assurez-vous que toutes les features sont présentes."}), 400
        
        # Faire la prédiction
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)[:, 1]
        
        # Retourner la réponse
        return jsonify({"prediction": prediction.tolist(), "probability": prediction_proba.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
