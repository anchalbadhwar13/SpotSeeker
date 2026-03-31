import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

def encode_cyclical_time(df, col, max_val):
    """Converts linear time into circular coordinates (Sine/Cosine)."""
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def train_model():
    print("1. Loading team dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'parking_data.csv') 
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}. Check filename.")
        return

    print("2. Preprocessing & Feature Engineering...")
    # Apply Time Science
    df = encode_cyclical_time(df, 'Hour', 24) 
    df = encode_cyclical_time(df, 'Day_of_Week', 7)

    # Encode categorical labels
    status_mapping = {'Empty': 0, 'Moderate': 1, 'Busy': 2, 'Full': 3}
    df['Occupancy_Status'] = df['Occupancy_Status'].map(status_mapping)
    df = pd.get_dummies(df, columns=['Weather_Condition', 'Parking_Zone_ID'])

    # Drop non-predictive or redundant columns
    cols_to_drop = ['Hour', 'Day_of_Week', 'Day_Name', 'Month', 'Occupancy_Rate', 
                    'Available_Binary', 'Occupancy_Status']
    
    X = df.drop(columns=cols_to_drop)
    y = df['Occupancy_Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("3. Splitting into Training and Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print("4. Training Optimized Multi-Layer Perceptron...")
    
    nn_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001,
        learning_rate='adaptive',
        learning_rate_init=0.015,
        max_iter=1500, 
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    nn_model.fit(X_train, y_train)

    print("5. Evaluating Model Performance...")
    y_pred = nn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Final Model Accuracy: {accuracy * 100:.2f}%")

    print("6. Saving Model and Scaler...")
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    

    joblib.dump(nn_model, os.path.join(models_dir, 'spotseeker_nn.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    print("   -> Success! Model artifacts stored in the models directory.")

if __name__ == "__main__":
    train_model()

