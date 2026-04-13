import numpy as np
import pandas as pd

np.random.seed(42)
N = 15000

zones = ['Zone_A_Downtown', 'Zone_B_University', 'Zone_C_Shopping',
         'Zone_D_Residential', 'Zone_E_Airport']
weather_options = ['Sunny', 'Cloudy', 'Overcast', 'Light Rain', 'Heavy Rain']

hours       = np.random.randint(0, 24, N)
days        = np.random.randint(0, 7, N)
months      = np.random.randint(1, 13, N)
zone_ids    = np.random.choice(zones, N)
weathers    = np.random.choice(weather_options, N, p=[0.30, 0.25, 0.20, 0.15, 0.10])
temps       = np.round(np.random.normal(12, 8, N), 1).clip(-20, 40)
durations   = np.random.choice([15,30,60,90,120,180,240,360,480], N)
spec_events = np.random.choice([0, 1], N, p=[0.85, 0.15])
precip      = np.where(np.isin(weathers, ['Light Rain', 'Heavy Rain']), 1, 0)

p_available = np.full(N, 0.65)

# Rush hours → less available
rush = ((hours >= 8) & (hours <= 9)) | ((hours >= 16) & (hours <= 18))
p_available[rush] -= 0.30

# Late night → more available
night = (hours >= 22) | (hours <= 5)
p_available[night] += 0.25

# Weekends → busier for shopping/downtown, quieter for university
weekend = (days >= 5)
shopping = (zone_ids == 'Zone_C_Shopping')
university = (zone_ids == 'Zone_B_University')
downtown = (zone_ids == 'Zone_A_Downtown')

p_available[weekend & shopping]    -= 0.20
p_available[weekend & downtown]    -= 0.15
p_available[weekend & university]  += 0.20

# Special events → much less available
p_available[spec_events == 1] -= 0.35

# Bad weather → more available (people don't drive out)
bad_weather = np.isin(weathers, ['Heavy Rain'])
p_available[bad_weather] += 0.10

# Airport zone → busy at all hours
airport = (zone_ids == 'Zone_E_Airport')
p_available[airport] -= 0.10

# Clip to valid probability range
p_available = np.clip(p_available, 0.05, 0.95)

# Sample actual availability
available_binary = np.where(
    np.random.uniform(0, 1, N) < p_available,
    'Available', 'Unavailable'
)

df_new = pd.DataFrame({
    'Hour':                 hours,
    'Day_of_Week':          days,
    'Month':                months,
    'Weather_Condition':    weathers,
    'Precipitation':        precip,
    'Special_Event':        spec_events,
    'Temperature_C':        temps,
    'Parking_Duration_Min': durations,
    'Parking_Zone_ID':      zone_ids,
    'Available_Binary':     available_binary,
})

# Show distribution of new data
print("New data class distribution:")
print(df_new['Available_Binary'].value_counts())
print(df_new['Available_Binary'].value_counts(normalize=True).round(2))

# Load existing dataset and append
try:
    df_existing = pd.read_csv('synthetic_parking_dataset.csv')
    # Only keep columns that match
    common_cols = [c for c in df_new.columns if c in df_existing.columns]
    df_combined = pd.concat([df_existing[common_cols], df_new[common_cols]], ignore_index=True)
    print(f"\n✅ Combined: {len(df_existing)} existing + {N} new = {len(df_combined)} total rows")
except FileNotFoundError:
    df_combined = df_new
    print(f"\n✅ No existing file found — created fresh dataset with {N} rows")

print("\nFinal combined class distribution:")
print(df_combined['Available_Binary'].value_counts())
print(df_combined['Available_Binary'].value_counts(normalize=True).round(2))

df_combined.to_csv('synthetic_parking_dataset.csv', index=False)
print("\n✅ Saved synthetic_parking_dataset.csv")
print("🎉 Now run: python3 data_prep.py → python3 train_nn.py → python3 spotseeker_cli.py")