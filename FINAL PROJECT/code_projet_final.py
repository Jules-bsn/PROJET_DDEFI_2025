"""
PROJET FINAL - Prédiction churns opérateurs téléphoniques
Dataset Churns https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services?resource=download
"""
#Dataset transac metaverse: https://www.kaggle.com/datasets/faizaniftikharjanjua/metaverse-financial-transactions-dataset

#pip install pandas numpy matplotlib seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('customer_churn_telecom_services.csv')  # chargement données

### PIPELINE
## Extract (exploration et analyse)
print("Colonnes disponibles :", df.columns.tolist())

print(df.head()) #aperçu de la df 
print(df.describe()) #donne les stat, moyennes, quartiles, min, max. 
print(df.isnull().sum()) # Vérification des valeurs manquantes
print(df.duplicated().sum()) # Vérification des doublons
for col in df.select_dtypes(include='object').columns: #quels valeurs possibles par catégorie
    print(f"\n{col}: {df[col].unique()}")

# Distribution graph churns
plt.figure(figsize=(6,4)) 
sns.countplot(x='Churn', data=df, palette='coolwarm')
plt.title('Répartition du Churn')
plt.show()

# distribution des frais / durée abonnement
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges'] 
for feature in numerical_features:
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution de {feature}')
    plt.show()

# Relation entre les variables catégorielles et le churn
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaperlessBilling', 'PaymentMethod']
for feature in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=feature, hue='Churn', data=df)
    plt.title(f'Churn par {feature}')
    plt.xticks(rotation=45)
    plt.show()

# Matrice de corrélation pour les charges et durée des abonnements
correlation = df[numerical_features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()

## Transform
# Nettoyage valeurs manquantes
important_columns = ['Contract', 'tenure', 'MonthlyCharges', 'PhoneService', 'InternetService']
df.dropna(subset=important_columns, inplace=True)

# Encodage des variables catégorielles en valeurs numériques
label_encoders = {} 
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Séparation des données en données d'entrainement et de test
X = df.drop(columns=['Churn']) 
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Load si nécessaire

### ML 
## Random forest
scaler = StandardScaler() # Normalisation des données
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Évaluation du modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title('Matrice de Confusion')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

## Régression logistique
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# évaluation régression logistique
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
