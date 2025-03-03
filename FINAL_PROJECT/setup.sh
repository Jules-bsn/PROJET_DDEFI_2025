#!/bin/bash

# Mise à jour des paquets
echo "Mise à jour des paquets..."
sudo apt update && sudo apt upgrade -y

# Création et activation d'un environnement virtuel
echo "Création et activation de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate

# Installation des dépendances
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installation terminée !"
