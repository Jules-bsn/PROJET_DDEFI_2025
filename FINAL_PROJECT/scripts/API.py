from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

# Charger le modèle entraîné
MODEL_PATH = "deployment/final_model.pkl"
print("🔹 Chargement du modèle...")
model = joblib.load(MODEL_PATH)
print("✅ Modèle chargé avec succès.")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    API permettant de faire des prédictions de churn à partir de nouvelles données client.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Aucune donnée reçue. Veuillez envoyer un JSON valide."}), 400
        
        df = pd.DataFrame(data)
        print("🔹 Données reçues :", df.head())
        
        # Vérification des colonnes attendues
        expected_features = model.feature_names_in_
        missing_features = [feature for feature in expected_features if feature not in df.columns]
        
        if missing_features:
            return jsonify({
                "error": "Données invalides. Certaines colonnes attendues sont manquantes.",
                "missing_features": missing_features
            }), 400
        
        # Prédiction des probabilités de churn
        predictions = model.predict_proba(df)[:, 1]
        
        return jsonify({"predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
