import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_parking_data(filepath='synthetic_parking_dataset.csv'):
    df = pd.read_csv(filepath)

    # Encode categorical columns so sklearn models can read them
    le = LabelEncoder()
    df['Zone_Encoded']    = le.fit_transform(df['Parking_Zone_ID'])
    df['Weather_Encoded'] = le.fit_transform(df['Weather_Condition'])

    # Features all models should use
    features = [
        'Hour', 'Day_of_Week', 'Month',
        'Zone_Encoded', 'Weather_Encoded',
        'Temperature_C', 'Precipitation', 'Special_Event'
    ]

    X = df[features]
    y = df['Occupancy_Status']  # 4-class: Empty/Moderate/Busy/Full

    print(f"Loaded: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Classes: {sorted(y.unique())}")
    return X, y

# Usage (one line to test it works):
X, y = load_parking_data('synthetic_parking_dataset.csv')