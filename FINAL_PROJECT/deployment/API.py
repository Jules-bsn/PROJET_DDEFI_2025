import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from pipeline import preprocess_data

# Charger le modèle entraîné
model_path = "deployment/final_model.pkl"
model = joblib.load(model_path)

# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour effectuer des prédictions sur de nouvelles données."""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        print("🔹 Prétraitement des données...")
        df = preprocess_data(df)
        
        # Vérification des colonnes manquantes
        missing_cols = set(model.feature_names_in_) - set(df.columns)
        if missing_cols:
            return jsonify({"error": f"Colonnes manquantes: {missing_cols}"}), 400
        
        df = df[model.feature_names_in_]
        prediction = model.predict_proba(df)[:, 1][0]  # Probabilité de churn
        
        return jsonify({"churn_probability": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
