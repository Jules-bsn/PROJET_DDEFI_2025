
import pandas as pd
import numpy as np
import os

from Pipeline import pre_traitement, pipeline

os.chdir('/Users/julesbesson/Documents/CENTRALE marseille/S9/PROJET_DDEFI_2025/FINAL_PROJECT')
dataframe_brut = pd.read_csv('customer_churn_telecom_services.csv') # chargement données

def Ajout_colonnes_feature_engineering(df):
    df = df.copy()

    # tenure_group : Regrouper l'ancienneté
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, np.inf],
                                labels=['0-12 mois', '12-24 mois', '24-48 mois', '48-72 mois', '72+ mois'])
    
    df['tenure_group'] = df['tenure_group'].fillna('0-12 mois')

    # avg_monthly_charge : Moyenne mensuelle (attention aux divisions par zéro)
    df['avg_monthly_charge'] = df.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
        axis=1
    )

    # is_long_term_contract : Contrat de 2 ans
    df['is_long_term_contract'] = (df['Contract'] == 'Two year').astype(int)

    # num_services : Nombre de services souscrits (hors contrat/facturation)
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['num_services'] = df[service_cols].apply(
        lambda row: sum(1 for x in row if x in ['Yes', 'Fiber optic']), axis=1
    )

    # has_security_package : Options de sécurité actives
    df['has_security_package'] = df[['OnlineSecurity', 'DeviceProtection']].apply(
        lambda row: int('Yes' in row.values), axis=1
    )

    # streaming_user : Utilise au moins un service de streaming
    df['streaming_user'] = df[['StreamingTV', 'StreamingMovies']].apply(
        lambda row: int('Yes' in row.values), axis=1
    )

    # engagement_score : Score simple d'engagement
    df['engagement_score'] = (
        df['tenure'] * 0.1 +
        df['num_services'] * 1 +
        df['is_long_term_contract'] * 5
    )

    # is_paperless_and_monthly : Mensuel + facturation dématérialisée
    df['is_paperless_and_monthly'] = (
        ((df['PaperlessBilling'] == 'Yes') & (df['Contract'] == 'Month-to-month')).astype(int)
    )

    # high_monthly_charge : Client qui paye plus que la moyenne
    monthly_charge_mean = df['MonthlyCharges'].mean()
    df['high_monthly_charge'] = (df['MonthlyCharges'] > monthly_charge_mean).astype(int)


    return df

def pipeline_with_feature_engineering(dataframe):
    df = dataframe.copy()
    df = pre_traitement(df)
    print('A = ', df.isna().sum())
    df = Ajout_colonnes_feature_engineering(df)
    print('B = ',df.isna().sum())
    df = pipeline(df)
    print('C = ',df.isna().sum())
    return df

'''
df = dataframe_brut.copy()
df = pipeline_with_feature_engineering(df)
print(df.isna().sum())'''