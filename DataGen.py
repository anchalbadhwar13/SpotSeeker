# note
# Parking_Zone_ID and Weather_Condition are string columns — sklearn models need them encoded first. 
# Either use LabelEncoder (shown above) or pd.get_dummies()

import pandas as pd

df = pd.read_csv('synthetic_parking_dataset.csv')

# print(df.shape)          # Should print (5000, 13)
# print(df.dtypes)         # Check column types look right
# print(df.isnull().sum()) # Should all be 0

# Features your models will use
features = ['Hour', 'Day_of_Week', 'Month', 'Parking_Zone_ID',
            'Weather_Condition', 'Temperature_C', 'Precipitation', 'Special_Event']

target_4class = df['Occupancy_Status']   # Decision Tree / KNN / Neural Net (4-class)
target_binary = df['Available_Binary']   # Minimal viable system
target_regression = df['Occupancy_Rate'] # If doing regression

# peak = df[df['Hour'].isin([8,9,10,17,18,19])]['Occupancy_Rate'].mean()
# offpeak = df[~df['Hour'].isin([8,9,10,17,18,19])]['Occupancy_Rate'].mean()
# weekday = df[df['Day_of_Week'] < 5]['Occupancy_Rate'].mean()
# weekend = df[df['Day_of_Week'] >= 5]['Occupancy_Rate'].mean()

# print(f"Peak: {peak:.2f} | Off-peak: {offpeak:.2f}")   # Peak should be much higher
# print(f"Weekday: {weekday:.2f} | Weekend: {weekend:.2f}") # Weekday should be higher

# For sklearn: encode categoricals
from sklearn.preprocessing import LabelEncoder
df['Zone_Enc'] = LabelEncoder().fit_transform(df['Parking_Zone_ID'])
df['Weather_Enc'] = LabelEncoder().fit_transform(df['Weather_Condition'])