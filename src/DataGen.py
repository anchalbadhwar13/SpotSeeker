import pandas as pd

df = pd.read_csv('synthetic_parking_dataset.csv')

# Features your models will use
features = ['Hour', 'Day_of_Week', 'Month', 'Parking_Zone_ID',
            'Weather_Condition', 'Temperature_C', 'Precipitation', 'Special_Event']

target_4class = df['occupancy_Status']   # Decision Tree / KNN / Neural Net (4-class)
target_binary = df['Available_Binary']   # Minimal viable system
target_regression = df['Occupancy_Rate'] # If doing regression

# For sklearn: encode categoricals
from sklearn.preprocessing import LabelEncoder
df['Zone_Enc'] = LabelEncoder().fit_transform(df['Parking_Zone_ID'])
df['Weather_Enc'] = LabelEncoder().fit_transform(df['Weather_Condition'])