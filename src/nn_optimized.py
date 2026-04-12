import pandas as pd
import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the CLEANED data
data_dir = 'processed_data'
X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()

print("Starting Neural Network Optimization")

# 2. Hyperparameter Optimization 
mlp_optimized = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), 
    activation='relu', 
    solver='adam', 
    alpha=0.0001, 
    max_iter=1000, 
    verbose=True,  # CRITICAL: This enables the log output
    random_state=42
)

# 3. Training
mlp_optimized.fit(X_train, y_train)

# 4. Final Evaluation (For Manya to compare)
y_pred = mlp_optimized.predict(X_test)
print(f"\nFinal Optimized Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Performance:\n", classification_report(y_test, y_pred))

# 5. Save the final model
joblib.dump(mlp_optimized, 'spotseeker_final_nn.joblib')
print("--- Final Optimized Model Saved ---")