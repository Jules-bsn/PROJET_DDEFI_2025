from pipeline import process_pipeline

# Chemins d'accès
RAW_DATA_PATH = 'data/raw/customer_churn_telecom_services.csv'
PROCESSED_DATA_PATH = 'data/processed/cleaned_data.csv'

# Exécution du pipeline de transformation des données
process_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)
