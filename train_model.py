import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Patch tqdm into sklearn for progress bar
from sklearn.model_selection import ParameterSampler
def tqdm_search(param_dist, n_iter, random_state=None):
    return list(tqdm(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state), desc="🔍 Searching"))

# Load enhanced dataset (with 18 features + Label)
df = pd.read_csv("enhanced_dataset.csv")

# Verify feature count
assert df.shape[1] == 19, f"Expected 19 columns (18 features + Label), got {df.shape[1]}"

# Split features and label
X = df.drop("Label", axis=1)
y = df["Label"]

# Verify number of features
assert X.shape[1] == 18, f"Expected 18 features, got {X.shape[1]}"

# Load and apply scaler
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Base model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

# Generate parameter list for display
param_list = tqdm_search(param_dist, n_iter=20, random_state=42)

# Randomized SearchCV
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=20, cv=5, verbose=0,
    random_state=42, n_jobs=1,  # n_jobs=1 is Windows safe
    scoring='f1_macro'
)

print("🚀 Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)
print(f"✅ Best parameters found: {random_search.best_params_}")

# Get best model and retrain on full training set
best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)

# Make predictions
y_pred = best_rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print(f"🎯 Macro F1 Score: {f1:.4f}")
print("\n📊 Classification Report:\n", report)

# Confusion Matrix Plot
plt.figure(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Away Win (-1)', 'Draw (0)', 'Home Win (1)'],
            yticklabels=['Away Win (-1)', 'Draw (0)', 'Home Win (1)'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(best_rf, "football_model_optimized.pkl")
print("\n💾 Model saved as 'football_model_optimized.pkl'")

# Save evaluation metrics
with open("model_evaluation.txt", "w", encoding='utf-8') as f:
    f.write(f"Best Parameters: {random_search.best_params_}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1 Score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)

print("✅ Evaluation metrics saved to 'model_evaluation.txt'")
