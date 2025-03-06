import os
import pandas as pd
from pipeline import process_pipeline

# Définition des chemins d'accès
RAW_DATA_PATH = "data/raw/customer_churn_telecom_services.csv"
PROCESSED_DATA_DIR = "data/processed"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_data.csv")

# Exécution du pipeline de nettoyage et transformation des données
process_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)
