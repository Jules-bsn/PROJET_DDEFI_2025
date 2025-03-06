import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor

def preprocess_data(df):
    """
    Effectue le prétraitement des données :
    - Nettoyage des colonnes
    - Remplacement des valeurs manquantes
    - Création de nouvelles variables
    - Standardisation des valeurs numériques
    - Transformation en variables indicatrices
    """
    df.columns = df.columns.str.strip()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Remplissage des valeurs manquantes
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())
    
    # Création de nouvelles variables
    df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['engagement_score'] = df['tenure'] * 0.2 + df['PaperlessBilling'].map({'Yes': 1, 'No': 0}) * 1.2 + df['Contract'].map({'Two year': 4, 'One year': 2, 'Month-to-month': 0})
    
    # Standardisation
    scaler = StandardScaler()
    df[['TotalCharges', 'avg_monthly_charge', 'engagement_score']] = scaler.fit_transform(df[['TotalCharges', 'avg_monthly_charge', 'engagement_score']])
    
    # Transformation en variables indicatrices
    df = pd.get_dummies(df, drop_first=True)
    
    # Suppression des colonnes inutiles
    df = df.drop(columns=['MonthlyCharges', 'num_services'], errors='ignore')
    
    return df

def apply_smote(X, y):
    """Applique la technique SMOTE pour équilibrer les classes."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def remove_high_vif_features(X, threshold=10):
    """
    Supprime progressivement les variables ayant un VIF élevé.
    """
    def calculate_vif(X):
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    
    vif_data = calculate_vif(X)
    while vif_data["VIF"].max() > threshold:
        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        print(f"Suppression de {feature_to_remove} (VIF={vif_data['VIF'].max():.2f})")
        X = X.drop(columns=[feature_to_remove])
        vif_data = calculate_vif(X)
    
    return X
