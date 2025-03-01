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
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('customer_churn_telecom_services.csv')  # chargement données

######### PIPELINE ########
#### Extract (exploration et analyse)
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

#### Transform
# Nettoyage valeurs manquantes dans colonnes importantes
important_columns = ['Contract', 'tenure', 'MonthlyCharges', 'PhoneService', 'InternetService']
df.dropna(subset=important_columns, inplace=True)

# convertit 'TotalCharges' en numérique + gère les valeurs manquantes
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encodage des variables catégorielles en valeurs numériques
label_encoders = {} 
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Dummies / one-hot encoding pour  variables catégorielles
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# normalisation variables numériques, assure que toutes les variables numériques sont à la même échelle
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Séparation des données en données d'entrainement et de test
X = df.drop(columns=['Churn']) 
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Matrice de corrélation var catégorielles
correlation_matrix = df.corr() # Calcul 
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Matrice de corrélation')
plt.show()

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#### Load si nécessaire

######### ML ########
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Régression logistique": LogisticRegression()
}

# Entraînement + évaluation des modèles
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\n{name} Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
    plt.title(f'Matrice de Confusion - {name}')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.show()

# Cross-validation pour chaque modèle
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # Cross-validation 5-fold
    cv_results[name] = scores.mean()  # Moyenne des scores

# résultats  cross-validation
print("\nRésultats de la Cross-Validation :")
for model, score in cv_results.items():
    print(f"{model}: {score:.4f}")

#### Facteurs de résiliation
importances = models["Random Forest"].feature_importances_
feature_names = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names, palette="coolwarm")
plt.title("Importance des variables pour le churn")
plt.show()

#### Segmentation clients à risque (10 plus à risque de churn)
df['Churn_Probability'] = models["Random Forest"].predict_proba(X)[:, 1]
df['Client_Index'] = df.index  # Utiliser l'index comme identifiant client

top_churners = df[['Client_Index', 'Churn_Probability', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges']]\
                .sort_values(by='Churn_Probability', ascending=False)\
                .head(10)

print("\nTop 10 clients les plus à risque de churn :")
print(top_churners)

#### Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

sns.pairplot(df, hue='Cluster', vars=['tenure', 'MonthlyCharges', 'TotalCharges'])
plt.show()

######### Contenairisation etc ########