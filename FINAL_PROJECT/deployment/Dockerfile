FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
COPY deployment/app.py ./
COPY deployment/gradient_boosting_model.pkl ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port Flask
EXPOSE 5000

# Commande pour exécuter l'API Flask
CMD ["python", "app.py"]
