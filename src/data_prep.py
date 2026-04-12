# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# df = pd.read_csv('synthetic_parking_dataset.csv')

# # 2. Define feature/target columns based on current dataset schema
# feature_cols = [
# 	'Hour',
# 	'Day_of_Week',
# 	'Month',
# 	'Weather_Condition',
# 	'Precipitation',
# 	'Special_Event',
# 	'Temperature_C',
# 	'Parking_Duration_Min'
# ]
# target_col = 'Available_Binary'

# missing = [c for c in feature_cols + [target_col] if c not in df.columns]
# if missing:
# 	raise ValueError(
# 		f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
# 	)

# X = df[feature_cols].copy()
# y = df[target_col].copy()

# # Encode categorical feature(s) before scaling
# X = pd.get_dummies(X, columns=['Weather_Condition'], drop_first=True)

# # 3. Perform 80/20 Train/Test Split 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Apply StandardScaler 
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# output_dir = 'processed_data'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f'{output_dir}/X_train.csv', index=False)
# pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f'{output_dir}/X_test.csv', index=False)
# y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
# y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

# print("Data Preprocessing Complete: 4 files exported.")
#!/usr/bin/env python3
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

# ── 1. Load raw dataset ───────────────────────────────────────────────────────
df = pd.read_csv('synthetic_parking_dataset.csv')
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Define features and target ─────────────────────────────────────────────
feature_cols = [
    'Hour',
    'Day_of_Week',
    'Month',
    'Weather_Condition',    # categorical → will be one-hot encoded
    'Precipitation',
    'Special_Event',
    'Temperature_C',
    'Parking_Duration_Min',
    'Parking_Zone_ID',      # categorical → will be one-hot encoded
]
target_col = 'Available_Binary'  # values: 'Available' / 'Unavailable'

# Validate columns exist
missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}\nFound: {list(df.columns)}")

X = df[feature_cols].copy()
y = df[target_col].copy()

# ── 3. One-hot encode categoricals ────────────────────────────────────────────
# drop_first=False so CLI can set any zone/weather column explicitly
X = pd.get_dummies(X, columns=['Weather_Condition', 'Parking_Zone_ID'], drop_first=False)

# Save exact column order — CLI MUST use this same order
FEATURE_COLUMNS = list(X.columns)
print(f"✅ Feature columns after encoding ({len(FEATURE_COLUMNS)}):")
for col in FEATURE_COLUMNS:
    print(f"   {col}")

# ── 4. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Split: {len(X_train)} train / {len(X_test)} test rows")

# ── 5. Fit scaler on train set only, then transform both ─────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 6. Save processed CSVs ────────────────────────────────────────────────────
os.makedirs('processed_data', exist_ok=True)
pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS).to_csv('processed_data/X_train.csv', index=False)
pd.DataFrame(X_test_scaled,  columns=FEATURE_COLUMNS).to_csv('processed_data/X_test.csv',  index=False)
y_train.to_csv('processed_data/y_train.csv', index=False)
y_test.to_csv( 'processed_data/y_test.csv',  index=False)
print("\n✅ Saved processed_data/ CSVs")

# ── 7. Save scaler and feature column list (used by CLI at runtime) ───────────
joblib.dump(scaler,          'spotseeker_scaler.joblib')
joblib.dump(FEATURE_COLUMNS, 'spotseeker_feature_cols.joblib')
print("✅ Saved spotseeker_scaler.joblib")
print("✅ Saved spotseeker_feature_cols.joblib")
print("\n🎉 Preprocessing complete. Now run: python3 train_nn.py")