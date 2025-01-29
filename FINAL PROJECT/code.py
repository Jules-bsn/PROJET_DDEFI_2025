#### PROJET FINAL - Analyse des comportements des investisseurs
#Dataset finance à trouver: Kaggle ? 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
import joblib

# Chargement des données (Exemple, à adapter selon Kaggle API)
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Prétraitement des données
def preprocess_data(data):
    # Suppression des valeurs manquantes
    data = data.dropna()
    
    # Sélection des variables pertinentes
    features = ['montant_transaction', 'frequence_transaction', 'produit_financier']
    X = data[features]
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Clustering avec K-Means
def apply_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

# Sauvegarde du modèle
def save_model(model, scaler, filename="models/kmeans_model.pkl"):
    joblib.dump((model, scaler), filename)
    
# Chargement du modèle
def load_model(filename="models/kmeans_model.pkl"):
    return joblib.load(filename)

# Visualisation avec PCA
def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='viridis')
    plt.title('Segmentation des investisseurs')
    plt.show()

# Pipeline complète
def pipeline(filepath):
    data = load_data(filepath)
    X_scaled, scaler = preprocess_data(data)
    kmeans, labels = apply_kmeans(X_scaled, 3)
    save_model(kmeans, scaler)
    plot_clusters(X_scaled, labels)

# Interface interactive avec Streamlit
def dashboard():
    st.title("Segmentation des investisseurs")
    data = load_data("transactions.csv")
    X_scaled, scaler = preprocess_data(data)
    kmeans, labels = apply_kmeans(X_scaled, 3)
    plot_clusters(X_scaled, labels)
    st.write("Clustering terminé et affiché ci-dessus.")
    
if __name__ == "__main__":
    dashboard()