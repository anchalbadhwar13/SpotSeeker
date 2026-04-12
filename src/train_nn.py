# import pandas as pd
# import numpy as np
# import os
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler 

# def encode_cyclical_time(df, col, max_val):
#     """Converts linear time into circular coordinates (Sine/Cosine)."""
#     df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
#     df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
#     return df

# def train_model():
#     print("1. Loading team dataset...")
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     data_path = os.path.join(script_dir, '..', 'data', 'parking_data.csv') 
    
#     try:
#         df = pd.read_csv(data_path)
#     except FileNotFoundError:
#         print(f"Error: File not found at {data_path}. Check filename.")
#         return

#     print("2. Preprocessing & Feature Engineering...")
#     # Apply Time Science
#     df = encode_cyclical_time(df, 'Hour', 24) 
#     df = encode_cyclical_time(df, 'Day_of_Week', 7)

#     # Encode categorical labels
#     status_mapping = {'Empty': 0, 'Moderate': 1, 'Busy': 2, 'Full': 3}
#     df['Occupancy_Status'] = df['Occupancy_Status'].map(status_mapping)
#     df = pd.get_dummies(df, columns=['Weather_Condition', 'Parking_Zone_ID'])

#     # Drop non-predictive or redundant columns
#     cols_to_drop = ['Hour', 'Day_of_Week', 'Day_Name', 'Month', 'Occupancy_Rate', 
#                     'Available_Binary', 'Occupancy_Status']
    
#     X = df.drop(columns=cols_to_drop)
#     y = df['Occupancy_Status']

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     print("3. Splitting into Training and Testing sets...")
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#     print("4. Training Optimized Multi-Layer Perceptron...")
    
#     nn_model = MLPClassifier(
#         hidden_layer_sizes=(512, 256, 128, 64), 
#         activation='relu', 
#         solver='adam', 
#         alpha=0.0001,
#         learning_rate='adaptive',
#         learning_rate_init=0.015,
#         max_iter=1500, 
#         random_state=42,
#         early_stopping=True,
#         validation_fraction=0.1
#     )
    
#     nn_model.fit(X_train, y_train)

#     print("5. Evaluating Model Performance...")
#     y_pred = nn_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"   -> Final Model Accuracy: {accuracy * 100:.2f}%")

#     print("6. Saving Model and Scaler...")
#     models_dir = os.path.join(script_dir, '..', 'models')
#     os.makedirs(models_dir, exist_ok=True)
    

#     joblib.dump(nn_model, os.path.join(models_dir, 'spotseeker_nn.joblib'))
#     joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
#     print("   -> Success! Model artifacts stored in the models directory.")

# if __name__ == "__main__":
#     train_model()

#!/usr/bin/env python3
"""
SpotSeeker — Neural Network Training (Fixed)
Uses class_weight balancing to fix the 76/24 imbalance problem.
Run this AFTER data_prep.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── 1. Load preprocessed scaled data ─────────────────────────────────────────
data_dir = 'processed_data'
X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv')).values
X_test  = pd.read_csv(os.path.join(data_dir, 'X_test.csv')).values
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
y_test  = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()

print(f"✅ Loaded training data: {X_train.shape[0]} rows, {X_train.shape[1]} features")

# Show class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n📊 Class distribution in training set:")
for cls, cnt in zip(unique, counts):
    pct = cnt / len(y_train) * 100
    print(f"   {cls}: {cnt} rows ({pct:.1f}%)")

# ── 2. Compute class weights manually ────────────────────────────────────────
# Tells the model to penalise mistakes on 'Unavailable' much more heavily.
# Formula: weight = total_samples / (n_classes * class_count)
n_samples = len(y_train)
n_classes = len(unique)
class_weights = {}
for cls, cnt in zip(unique, counts):
    class_weights[cls] = n_samples / (n_classes * cnt)

print(f"\n⚖️  Class weights applied:")
for cls, w in class_weights.items():
    print(f"   {cls}: {w:.3f}")

# MLPClassifier uses sample_weight per row, not class_weight directly
sample_weights = np.array([class_weights[label] for label in y_train])

# ── 3. Train model ────────────────────────────────────────────────────────────
print("\nTraining Neural Network with class balancing...")
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

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Final Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Detailed Performance:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred, labels=['Available', 'Unavailable'])
print(f"                       Pred:Available  Pred:Unavailable")
print(f"Actual:Available            {cm[0][0]:<16}{cm[0][1]}")
print(f"Actual:Unavailable          {cm[1][0]:<16}{cm[1][1]}")

# ── 5. Save model ─────────────────────────────────────────────────────────────
joblib.dump(model, 'spotseeker_final_nn.joblib')
print("\n✅ Saved spotseeker_final_nn.joblib")
print("🎉 Training complete. Now run: python3 spotseeker_cli.py")