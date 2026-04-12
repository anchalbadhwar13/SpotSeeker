# SpotSeeker — Neural Network Training 

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configuration for paths relative to project root
DATA_DIR = 'processed_data'
MODEL_DIR = 'models'

# Ensure the models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load preprocessed and scaled datasets
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv')).values
X_test  = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv')).values
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()

print(f"Dataset Loaded: {X_train.shape[0]} training samples found.")

# Calculate class weights to address data imbalance
# Formula: weight = total_samples / (n_classes * class_count)
unique, counts = np.unique(y_train, return_counts=True)
n_samples = len(y_train)
n_classes = len(unique)
class_weights = {cls: n_samples / (n_classes * cnt) for cls, cnt in zip(unique, counts)}

# Map weights to training labels
sample_weights = np.array([class_weights[label] for label in y_train])

print("Training Multi-layer Perceptron (MLP) model...")

# Model parameters based on optimized performance testing [cite: 137]
# Architecture: 256 -> 128 -> 64 hidden nodes
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=False,
    validation_fraction=0.15,
    n_iter_no_change=20,
    verbose=True
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Performance Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['Available', 'Unavailable'])
print(f"                       Pred:Available  Pred:Unavailable")
print(f"Actual:Available            {cm[0][0]:<16}{cm[0][1]}")
print(f"Actual:Unavailable          {cm[1][0]:<16}{cm[1][1]}")

# Export final model for CLI and production use
model_filename = os.path.join(MODEL_DIR, 'spotseeker_final_nn.joblib')
joblib.dump(model, model_filename)

print(f"\nModel saved successfully to {model_filename}")
print("System ready for inference via spotseeker_cli.py")