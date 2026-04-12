import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# 1. Load the model
model = joblib.load("spotseeker_final_nn.joblib")

# 2. Load the data
X_test = pd.read_csv("X_test.csv")

# 3. FIX: Scale the data before predicting
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test) 

# 4. Predict using the scaled data
y_pred = model.predict(X_test_scaled)

# 5. Save
pd.DataFrame(y_pred, columns=["y_pred"]).to_csv("nn_predictions.csv", index=False)
print("✅ Re-generated predictions with scaling!")