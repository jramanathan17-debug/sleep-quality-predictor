import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("dataset/sleep_data.csv")

print("Dataset loaded successfully!")

# 2. Convert sleep quality score into categories
def categorize_sleep(score):
    if score <= 5:
        return "Poor"
    elif score <= 7:
        return "Average"
    else:
        return "Good"

data["Sleep Quality"] = data["Quality of Sleep"].apply(categorize_sleep)

# 3. Select input features and target
X = data.drop(
    columns=["Person ID", "Quality of Sleep", "Sleep Quality"]
)
y = data["Sleep Quality"]

# 4. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# 5. Prepare categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# 6. Create Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train the model
print("Training model...")
model.fit(X_train, y_train)

# 9. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# 10. Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sleep_quality_model.pkl")

print("Model saved successfully!")
print("Location: models/sleep_quality_model.pkl")