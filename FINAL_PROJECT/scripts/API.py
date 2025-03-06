from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pipeline import prepare_prediction_data, clean_data, remove_multicollinearity, normalize_features, ensure_all_columns

app = Flask(__name__)

# Charger le modèle pré-entraîné
model = joblib.load('deployment/final_model.pkl')

# Charger un DataFrame de référence pour s'assurer que toutes les colonnes sont présentes
reference_df = pd.read_csv('data/processed/cleaned_data.csv')
reference_df = clean_data(reference_df)
reference_df = remove_multicollinearity(reference_df)
reference_df = normalize_features(reference_df)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Prétraitement des données
    df = prepare_prediction_data(data, reference_df)
    
    # Prédiction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)[:, 1]
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
