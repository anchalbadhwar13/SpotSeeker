"""
SpotSeeker — Data Preprocessing
Cleans, encodes, scales, and saves all artifacts needed by the CLI.
Run this FIRST before train_nn.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('synthetic_parking_dataset.csv')
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

feature_cols = [
    'Hour',
    'Day_of_Week',
    'Month',
    'Weather_Condition',    
    'Precipitation',
    'Special_Event',
    'Temperature_C',
    'Parking_Duration_Min',
    'Parking_Zone_ID',      
]
target_col = 'Available_Binary'  # values: 'Available' / 'Unavailable'

# Validate columns exist
missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}\nFound: {list(df.columns)}")

X = df[feature_cols].copy()
y = df[target_col].copy()

X = pd.get_dummies(X, columns=['Weather_Condition', 'Parking_Zone_ID'], drop_first=False)

FEATURE_COLUMNS = list(X.columns)
print(f"✅ Feature columns after encoding ({len(FEATURE_COLUMNS)}):")
for col in FEATURE_COLUMNS:
    print(f"   {col}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Split: {len(X_train)} train / {len(X_test)} test rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

os.makedirs('processed_data', exist_ok=True)
pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS).to_csv('processed_data/X_train.csv', index=False)
pd.DataFrame(X_test_scaled,  columns=FEATURE_COLUMNS).to_csv('processed_data/X_test.csv',  index=False)
y_train.to_csv('processed_data/y_train.csv', index=False)
y_test.to_csv( 'processed_data/y_test.csv',  index=False)
print("\n✅ Saved processed_data/ CSVs")


joblib.dump(scaler,          'spotseeker_scaler.joblib')
joblib.dump(FEATURE_COLUMNS, 'spotseeker_feature_cols.joblib')
print("✅ Saved spotseeker_scaler.joblib")
print("✅ Saved spotseeker_feature_cols.joblib")
print("\n🎉 Preprocessing complete. Now run: python3 train_nn.py")