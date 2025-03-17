import numpy as np
import pandas as pd
import joblib # For saving and loading the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data # Features
y = data.target # Labels

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Model saved as random_forest_model.pkl")

# Load the saved model
loaded_model = joblib.load("random_forest_model.pkl")

# Test it with a sample input
sample_input = [X_test[0]] # Using a sample test data
predicted_class = loaded_model.predict(sample_input)
print(f"Predicted class: {predicted_class[0]}")