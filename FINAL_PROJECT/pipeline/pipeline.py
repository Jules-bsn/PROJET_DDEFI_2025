import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    df = df.copy()
    
    # Nettoyage des données
    df.columns = df.columns.str.strip()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)
    
    # Encodage des variables catégoriques
    categorical_features = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Normalisation des colonnes clés
    scaler = StandardScaler()
    cols_to_scale = ['TotalCharges', 'avg_monthly_charge']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df

def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
