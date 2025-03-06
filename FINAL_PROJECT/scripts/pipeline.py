import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(file_path):
    """Charge les données depuis un fichier CSV."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    """Nettoie et transforme les données."""
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())
    
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
    df = df.drop(columns=['MonthlyCharges', 'num_services'], errors='ignore')
    
    return df

def balance_data(X, y):
    """Applique SMOTE pour équilibrer les classes."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def remove_high_vif_features(X):
    """Supprime les variables avec un VIF trop élevé (>10)."""
    def calculate_vif(X):
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    X = X.astype('float64')
    
    vif_data = calculate_vif(X)
    while vif_data["VIF"].max() > 10:
        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        X = X.drop(columns=[feature_to_remove])
        vif_data = calculate_vif(X)
    
    return X
