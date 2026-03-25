import pandas as pd
import numpy as np
import os

def generate_dummy_data(num_rows=1000):
    np.random.seed(42)

    hours = np.random.randint(0, 24, num_rows)
    days = np.random.randint(0, 7, num_rows)
    weather = np.random.randint(0, 3, num_rows) # 0: sunny, 1: rain, 2: snow

    occupancy = np.random.randint(0, 4, num_rows) #0: empty, 1: moderate, 3: busy, 4: full
    df = pd.DataFrame({
        'hour': hours,
        'day_of_week': days,
        'weather_code': weather,
        'occupancy_class': occupancy
    })

    os.makedirs('data', exist_ok=True)

    file_path = 'data/synthetic_parking_data.csv'
    df.to_csv(file_path, index=False)
    print(f"Success! Generated {num_rows} rows of dummy data at {file_path}")

if __name__ == "__main__":
    generate_dummy_data()