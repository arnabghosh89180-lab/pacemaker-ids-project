import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("pacemaker_ids_project/datasets.csv")

print("✅ Dataset Loaded Successfully")
print(df.head())
print("\nColumns:", df.columns)

# ===============================
# TARGET COLUMN
# ===============================
target_column = "Type of attack"

if target_column not in df.columns:
    raise Exception("❌ Target column not found")

# Convert target into binary (Attack vs Normal)
y = df[target_column].apply(lambda x: 0 if "No Attack" in str(x) else 1)

# ===============================
# FEATURE SELECTION
# ===============================
X = df.drop(columns=[target_column])

# Keep only numeric columns
X = X.select_dtypes(include=['number'])

print("\nFeatures used:", X.columns)

# ===============================
# HANDLE MISSING VALUES
# ===============================
X = X.fillna(X.mean())

# ===============================
# TRAIN TEST SPLIT
# ===============================
if len(X) < 10:
    raise Exception("❌ Dataset too small")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# PREDICTION
# ===============================
y_pred = model.predict(X_test)

# ===============================
# METRICS
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ===============================
# PRINT RESULTS
# ===============================
print("\n📊 MODEL PERFORMANCE")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, "model.pkl")
print("\n✅ Model saved as model.pkl")