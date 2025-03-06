from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
from pipeline import preprocess_data

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
        
        df = pd.DataFrame([data])  # Convertir la ligne unique en DataFrame
        print("🔹 Données reçues avant traitement :", df.head())
        
        # Vérifier et supprimer 'Churn' s'il est présent
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])
        
        # S'assurer que toutes les colonnes sont des chaînes de caractères avant traitement
        df.columns = df.columns.astype(str)
        
        # Vérifier la présence de 'TotalCharges' et le convertir en float si nécessaire
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Appliquer le même traitement que dans data_cleaner
        df = preprocess_data(df)
        print("🔹 Données après prétraitement :", df.head())
        
        # Vérifier si toutes les colonnes attendues par le modèle sont présentes
        expected_features = model.feature_names_in_
        missing_features = [feature for feature in expected_features if feature not in df.columns]
        if missing_features:
            return jsonify({
                "error": "Données invalides. Certaines colonnes attendues sont manquantes après le prétraitement.",
                "missing_features": missing_features
            }), 400
        
        # Ajouter les colonnes manquantes avec des valeurs par défaut (0)
        for feature in missing_features:
            df[feature] = 0
        
        # Réordonner les colonnes pour correspondre au modèle
        df = df[expected_features]
        
        # Prédiction des probabilités de churn
        predictions = model.predict_proba(df)[:, 1]
        
        return jsonify({"predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
