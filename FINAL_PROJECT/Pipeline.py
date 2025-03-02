"""
PROJET FINAL - Prédiction churns opérateurs téléphoniques
Dataset Churns https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services?resource=download
"""
#Dataset transac metaverse: https://www.kaggle.com/datasets/faizaniftikharjanjua/metaverse-financial-transactions-dataset

#pip install pandas numpy matplotlib seaborn sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
import os

#__

#__
print(os.getcwd())
os.chdir('/Users/julesbesson/Documents/CENTRALE marseille/S9/PROJET_DDEFI_2025/FINAL_PROJECT')
dataframe_brut = pd.read_csv('customer_churn_telecom_services.csv') # chargement données

def pre_traitement(dataframe):
     
    df = dataframe.copy()
     
    colonnes_requises = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                         'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                         'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                         'MonthlyCharges', 'TotalCharges', 'Churn']
     
    colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
    if colonnes_manquantes:
        raise ValueError(f"Le dataframe ne contient pas les colonnes requises suivantes : {colonnes_manquantes}")
     
    colonne_categorielle = df.select_dtypes(include='object').columns
    colonne_numerique = df.select_dtypes(include=['number']).columns

      # Vérification du taux de valeurs manquantes
    for col in colonne_numerique:
        na_percent = (df[col].isna().sum() / len(df)) * 100
        if na_percent > 10:
            print(f"  **ALERTE**: La colonne '{col}' a un taux de valeurs manquantes de {na_percent:.2f}%, ce qui dépasse le seuil de 10%.")
            print(f"           -> Pensez à examiner attentivement cette colonne et potentiellement ajuster la stratégie d'imputation.")

    for col in colonne_categorielle:
        na_percent = (df[col].isna().sum() / len(df)) * 100
        if na_percent > 10:
            print(f"  **ALERTE**: La colonne '{col}' a un taux de valeurs manquantes de {na_percent:.2f}%, ce qui dépasse le seuil de 10%.")
            print(f"           -> Pensez à examiner attentivement cette colonne et potentiellement ajuster la stratégie d'imputation.")
    for col in colonne_categorielle : 
         df[col].fillna(df[col].mode()[0], inplace=True)

    for col in colonne_numerique :
        df[col].fillna(df[col].median(), inplace=True)

    return df 

# Fonction pour vérifier les valeurs manquantes dans chaque colonne
def verifier_valeurs_manquantes(df):
    colonnes_manquantes = False
    print("Vérification des valeurs manquantes par colonne :")
    for col in df.columns:
        na_percent = (df[col].isna().sum() / len(df)) * 100
        if na_percent > 0:  # Affiche seulement les colonnes avec des valeurs manquantes
            print(f"  - Colonne '{col}' : {na_percent:.2f}% de valeurs manquantes")
            colonnes_manquantes = True
    
    if not colonnes_manquantes:
        print("  - Aucune valeur manquante trouvée dans les colonnes.")

# Fonction pour préparer le dataframe pour les prédictions 

def encoder(dataframe):
    df = dataframe.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True) # Dropfirst --> 1 seule colonne pour la colonne avec 2 variables différentes  
    return df

# Fonction de vérification (avant et après l'encodage)
def verification(dataframe, phase="avant"):
    # Vérifie s'il y a encore des valeurs manquantes
    missing_values = dataframe.isna().sum().sum()
    if missing_values > 0:
        raise ValueError(f"**ALERTE**: Des valeurs manquantes ont été trouvées {phase} l'encodage.")
    
    # Vérifie qu'il y a bien des colonnes catégorielles avant l'encodage
    if phase == "avant":
        categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) == 0:
            raise ValueError("**ALERTE**: Aucune colonne catégorielle à traiter avant l'encodage.")
    
    return True


def pipeline(dataframe):
    df = dataframe.copy()
    df = pre_traitement(df) # Pré-traitement des données
    verification(df, phase="avant") # Vérification avant l'encodage
    df = encoder(df) # Encodage des variables catégorielles
    verification(df, phase="après") # Vérification après l'encodage
    return df


# A APPLIQUER seulement sur les data d'entrainement
def preprocess_data(df): # Utile seulement pour les modèles : Régression logistique, SVM, KNN, Réseaux de neurones, techniques de réduction de dimension
    """
    Fonction de prétraitement des données :
    - Identifie les colonnes numériques et dummies
    - Applique la normalisation appropriée
    - Retourne le DataFrame transformé
    """
    df = df.copy()  # Pour éviter de modifier l'original

    # Séparation des colonnes numériques et dummies --> je n'ai pas adapté cette fonction avec les features engineering supplémentaires
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    dummy_cols = [col for col in df.columns if df[col].nunique() == 2]  # Colonnes binaires (0/1)

    # Vérifier et convertir les valeurs numériques (cas où TotalCharges aurait des valeurs manquantes ou en str)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)

    # Normalisation selon la distribution observée en data-vis
    scalers = {
        "tenure": MinMaxScaler(),  # Car valeurs bornées (0-72)
        "MonthlyCharges": StandardScaler(),  # Distribution plus large
        "TotalCharges": PowerTransformer(method='yeo-johnson')  
    }

    for col, scaler in scalers.items():
        df[col] = scaler.fit_transform(df[[col]])

    return df

# Code chat GPT pour la généralisation : A TESTER
def preprocess_data_V2(df):
    """
    Fonction de prétraitement des données :
    - Identifie les colonnes numériques à normaliser
    - Identifie les colonnes binaires et catégorielles
    - Applique la normalisation et l'encodage appropriés
    - Retourne le DataFrame transformé
    """
    df = df.copy()  # Pour éviter de modifier l'original

    # Séparation des colonnes numériques, binaires et catégorielles
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()  # Colonnes numériques
    dummy_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'Churn']  # Colonnes binaires (0/1)
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in dummy_cols]  # Colonnes catégorielles

    # Suppression des colonnes trop spécifiques (par exemple, la colonne cible 'Churn')
    if 'Churn' in num_cols:
        num_cols.remove('Churn')
    
    # Normalisation des colonnes numériques
    scalers = {
        "MinMaxScaler": MinMaxScaler(),  # Colonnes bornées entre un minimum et un maximum (ex: tenure, num_services)
        "StandardScaler": StandardScaler(),  # Colonnes avec une distribution proche de normale (ex: MonthlyCharges)
        "PowerTransformer": PowerTransformer(method='yeo-johnson')  # Colonnes avec une distribution asymétrique (ex: TotalCharges)
    }
    
    # Normaliser les colonnes numériques
    for col in num_cols:
        if df[col].std() != 0:  # Si la variance n'est pas nulle
            # Appliquer un scaler en fonction des caractéristiques de la distribution
            if df[col].min() >= 0 and df[col].max() <= 100:  # Exemple : bornées entre 0 et 100
                df[col] = scalers["MinMaxScaler"].fit_transform(df[[col]])
            elif df[col].max() - df[col].min() > 100:  # Par exemple : des variables comme MonthlyCharges
                df[col] = scalers["StandardScaler"].fit_transform(df[[col]])
            else:  # Si les données sont asymétriques
                df[col] = scalers["PowerTransformer"].fit_transform(df[[col]])

    # Encodage des colonnes binaires (0/1)
    for col in dummy_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})  # Si nécessaire, map les valeurs 'Yes' et 'No' en 1 et 0

    # Encodage des colonnes catégorielles (si nécessaire, OneHotEncoding ou LabelEncoding)
    for col in cat_cols:
        df = pd.get_dummies(df, columns=[col], drop_first=True)  # OneHotEncoding

    return df


'''
df = dataframe_brut.copy()
df = pre_traitement(df)
df = Ajout_colonnes_feature_engineering(df)
df = pre_traitement(df)

print(df.isna().sum())
# Filtrer les lignes où 'tenure_groupe' est NaN
lignes_na = df[df['tenure_group'].isna()]['tenure']

# Afficher les lignes
print(lignes_na)'''