from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pipeline import clean_data, remove_multicollinearity, normalize_features

app = Flask(__name__)

# Charger le modèle pré-entraîné
model = joblib.load('deployment/final_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    
    # Prétraitement des données
    df = clean_data(df)
    df = remove_multicollinearity(df)
    df = normalize_features(df)
    
    # Prédiction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)[:, 1]
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
