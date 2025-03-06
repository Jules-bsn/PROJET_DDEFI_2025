import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Charge le dataset depuis le fichier spécifié."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def clean_data(df):
    """Nettoie et transforme les données pour le modèle."""
    df.drop_duplicates(inplace=True)
    
    # Gestion des valeurs manquantes uniquement sur les colonnes numériques
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())
    
    # Conversion de la colonne cible 'Churn' en valeurs numériques
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Création de nouvelles features
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # Évite division par zéro
    if 'tenure' in df.columns and 'PaperlessBilling' in df.columns and 'Contract' in df.columns:
        df['engagement_score'] = df['tenure'] * 0.2 + df['PaperlessBilling'].map({'Yes': 1, 'No': 0}) * 1.2 + df['Contract'].map({'Two year': 4, 'One year': 2, 'Month-to-month': 0})
    if 'Contract' in df.columns:
        df['is_long_term_contract'] = (df['Contract'] == 'Two year').astype(int)
    if all(col in df.columns for col in ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']):
        df['num_services'] = df[['PhoneService', 'MultipleLines', 'InternetService', 
                                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                 'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
            lambda row: sum(1 for x in row if x in ['Yes', 'Fiber optic']), axis=1
        )
    
    # Suppression des colonnes non pertinentes
    drop_columns = [
        'CustomerID', 'gender', 'PhoneService', 'tenure', 'MonthlyCharges',
        'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
        'StreamingMovies_No internet service', 'StreamingTV_No internet service',
        'TechSupport_No internet service', 'DeviceProtection_No internet service',
        'InternetService_No'
    ]
    df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore', inplace=True)
    
    # Encodage des variables catégoriques
    df = pd.get_dummies(df, drop_first=False)
    
    return df

def remove_multicollinearity(df, threshold=10.0):
    """Supprime les variables fortement colinéaires en utilisant le VIF."""
    
    df = df.select_dtypes(include=[np.number])  # On garde seulement les colonnes numériques
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    
    # 🔹 Supprimer les colonnes constantes
    nunique = df.nunique()
    constant_columns = nunique[nunique == 1].index.tolist()
    if constant_columns:
        print(f"⚠️ Suppression des colonnes constantes : {constant_columns}")
        df.drop(columns=constant_columns, inplace=True)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    while vif_data["VIF"].max() > threshold:
        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        print(f"⚠️ Suppression de la variable fortement colinéaire : {feature_to_remove} (VIF={vif_data['VIF'].max():.2f})")
        df.drop(columns=[feature_to_remove], inplace=True)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return df

def normalize_features(df):
    """Normalise les variables numériques clés, si elles existent dans le DataFrame."""
    scaler = StandardScaler()
    columns_to_scale = ['TotalCharges', 'avg_monthly_charge', 'num_services', 'engagement_score']
    existing_columns = [col for col in columns_to_scale if col in df.columns]
    if existing_columns:
        df[existing_columns] = scaler.fit_transform(df[existing_columns])
    return df

def balance_classes(X, y):
    """Applique SMOTE pour équilibrer les classes."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def process_pipeline(file_path, output_path):
    """Exécute le pipeline de transformation et sauvegarde les données nettoyées."""
    df = load_data(file_path)
    df = clean_data(df)
    df = remove_multicollinearity(df)
    df = normalize_features(df)
    
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    X, y = balance_classes(X, y)
    
    df_resampled = pd.DataFrame(X, columns=X.columns)
    df_resampled['Churn'] = y
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_resampled.to_csv(output_path, index=False)
    print(f"✅ Fichier nettoyé et équilibré sauvegardé : {output_path}")
