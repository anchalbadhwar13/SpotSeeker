import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv('synthetic_parking_dataset.csv')

# 2. Define feature/target columns based on current dataset schema
feature_cols = [
	'Hour',
	'Day_of_Week',
	'Month',
	'Weather_Condition',
	'Precipitation',
	'Special_Event',
	'Temperature_C',
	'Parking_Duration_Min'
]
target_col = 'Available_Binary'

missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
	raise ValueError(
		f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
	)

X = df[feature_cols].copy()
y = df[target_col].copy()

# Encode categorical feature(s) before scaling
X = pd.get_dummies(X, columns=['Weather_Condition'], drop_first=True)

# 3. Perform 80/20 Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply StandardScaler 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Export as separate CSVs for the team 
output_dir = 'processed_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f'{output_dir}/X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f'{output_dir}/X_test.csv', index=False)
y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

print("Data Preprocessing Complete: 4 files exported.")