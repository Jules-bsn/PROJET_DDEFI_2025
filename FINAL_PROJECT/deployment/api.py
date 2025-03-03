from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Charger le modèle
try:
    model = joblib.load("deployment/final_model.pkl")
    print("Modèle chargé avec succès !")
except Exception as e:
    print("Erreur lors du chargement du modèle :", str(e))
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    try:
        data = request.get_json()
        
        # Vérifier que les données sont bien reçues
        if not isinstance(data, list):
            return jsonify({"error": "Les données doivent être une liste de dictionnaires"}), 400
        
        df = pd.DataFrame(data)
        
        # Vérifier que toutes les colonnes nécessaires sont présentes
        expected_features = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                             "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
                             "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
                             "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
                             "MonthlyCharges", "TotalCharges"]
        
        missing_features = [col for col in expected_features if col not in df.columns]
        if missing_features:
            return jsonify({"error": "Colonnes manquantes", "missing_features": missing_features}), 400
        
        # Appliquer les mêmes transformations que lors de l'entraînement du modèle
        df = preprocess_data(df)
        
        # Prédiction
        prediction = model.predict(df)
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        print("Erreur lors de la prédiction :", str(e))
        print(traceback.format_exc())
        return jsonify({"error": "Erreur interne du serveur"}), 500

def preprocess_data(df):
    """Applique les mêmes transformations que dans clean_data.py"""
    # Exemple de transformations (à adapter selon clean_data.py)
    df = df.copy()
    
    # Convertir les colonnes numériques si elles sont mal typées
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(float)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
    
    # Encodage des variables catégorielles (simplifié)
    df = pd.get_dummies(df, drop_first=True)
    
    return df

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
