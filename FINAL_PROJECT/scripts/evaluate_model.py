import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def preprocess_data(df, feature_names):
    """Nettoie et transforme les données pour correspondre au modèle entraîné."""
    df = df.copy()
    
    # Vérifier et convertir 'TotalCharges' en numérique (peut contenir des valeurs vides)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['engagement_score'] = (
        df['tenure'] * 0.2 +
        df['PaperlessBilling'].map({'Yes': 1, 'No': 0}) * 1.2 +
        df['Contract'].map({'Two year': 4, 'One year': 2, 'Month-to-month': 0})
    )
    
    scaler = StandardScaler()
    df[['TotalCharges', 'avg_monthly_charge', 'engagement_score']] = scaler.fit_transform(
        df[['TotalCharges', 'avg_monthly_charge', 'engagement_score']]
    )
    
    df = pd.get_dummies(df, drop_first=True)
    
    # Assurer que toutes les colonnes du modèle sont présentes
    missing_cols = set(feature_names) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # Ajouter les colonnes manquantes avec des valeurs nulles
    
    # Réordonner les colonnes pour correspondre exactement à celles du modèle
    df = df[feature_names]
    
    return df.astype(np.float64)  # Conversion explicite pour éviter les erreurs de sérialisation

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
        df = preprocess_data(df, model.feature_names_in_)
        
        probability = model.predict_proba(df)[:, 1][0]  # Probabilité de churn
        prediction = "Yes" if probability >= 0.5 else "No"  # Seuil à 0.5 pour classification
        
        return jsonify({"churn_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """Évalue les performances du modèle sur un jeu de test et détecte l'overfitting."""
    try:
        # Charger les données traitées
        data_path = "data/processed/cleaned_data.csv"
        df = pd.read_csv(data_path)
        
        y = df['Churn']
        X = df.drop(columns=['Churn'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Évaluer sur le jeu de test
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        overfitting_status = "Yes" if train_score - test_score > 0.1 else "No"
        
        metrics = {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "overfitting_detected": overfitting_status,
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
        }
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
