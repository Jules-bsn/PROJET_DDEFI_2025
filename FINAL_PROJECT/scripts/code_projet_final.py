import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv("data/customer_churn_telecom_services.csv")

# Data Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop(columns=['Churn_Yes'])  # Adjust this column name as necessary
y = df['Churn_Yes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the final Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Predictions
y_pred = gb_model.predict(X_test)
y_prob = gb_model.predict_proba(X_test)[:, 1]

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Final Gradient Boosting Model – Performance:\nAccuracy: {accuracy:.4f}\nROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
feature_importance = gb_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_names, palette="coolwarm")
plt.title("Feature Importance for Churn Prediction - Gradient Boosting")
plt.show()

# Save the model
joblib.dump(gb_model, "deployment/gradient_boosting_model.pkl")
print("Optimized Gradient Boosting model saved!")
