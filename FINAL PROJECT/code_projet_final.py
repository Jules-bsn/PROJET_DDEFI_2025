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

# Exploration et analyse
print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum()) # Vérification des valeurs manquantes

plt.figure(figsize=(6,4)) # Graph churns
sns.countplot(x='Churn', data=df, palette='coolwarm')
plt.title('Répartition du Churn')
plt.show()

#Pipeline
label_encoders = {} # Transformation des données
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['Churn']) # Séparation des données
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modele machine learning: random forest
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