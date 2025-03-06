import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model_path):
    """
    Vérifie l'overfitting en comparant les scores de validation croisée et affiche des métriques supplémentaires.
    """
    print("\n🔹 Chargement du modèle XGBoost...")
    model = joblib.load(model_path)
    
    if hasattr(model, 'cv_results_'):
        train_score = np.mean(model.cv_results_['mean_train_score'])
        val_score = np.mean(model.cv_results_['mean_test_score'])
        
        print(f"✅ ROC-AUC moyen (train) : {train_score:.4f}")
        print(f"✅ ROC-AUC moyen (validation) : {val_score:.4f}")

        if train_score - val_score > 0.05:
            print("⚠️ Attention : Possible sur-apprentissage !")
        
        # Ajout des métriques complémentaires
        train_precision = np.mean(model.cv_results_.get('mean_train_precision', [None]))
        val_precision = np.mean(model.cv_results_.get('mean_test_precision', [None]))
        train_recall = np.mean(model.cv_results_.get('mean_train_recall', [None]))
        val_recall = np.mean(model.cv_results_.get('mean_test_recall', [None]))
        train_f1 = np.mean(model.cv_results_.get('mean_train_f1', [None]))
        val_f1 = np.mean(model.cv_results_.get('mean_test_f1', [None]))

        if train_precision and val_precision:
            print(f"✅ Précision (train) : {train_precision:.4f}, (validation) : {val_precision:.4f}")
        if train_recall and val_recall:
            print(f"✅ Rappel (train) : {train_recall:.4f}, (validation) : {val_recall:.4f}")
        if train_f1 and val_f1:
            print(f"✅ Score F1 (train) : {train_f1:.4f}, (validation) : {val_f1:.4f}")
        
    else:
        print("⚠️ Aucune validation croisée enregistrée dans le modèle.")

if __name__ == "__main__":
    evaluate_model("deployment/final_model.pkl")
