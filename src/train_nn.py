import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def encode_cyclical_time(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)

    return df

def train_model():
    print("1. Loading Synthetic Data...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'synthetic_parking_data.csv')

    df = pd.read_csv(data_path)
    
    print("2. Applying Temporal Engineering...")

    df = encode_cyclical_time(df, 'hour', 24)
    df = encode_cyclical_time(df, 'day_of_week', 7)

    X = df.drop(columns=['hour', 'day_of_week', 'occupancy_class'])
    y = df['occupancy_class']

    print("3. Splitting data into Training and Testing...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

    print("4. Initializing and Training Neural Network...")

    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state = 42)
    nn_model.fit(X_train, y_train)

    print("5. Evaluating the model...")

    y_pred = nn_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Baseline Accuracy on Dummy Data: {accuracy * 100:.2f}%")

    print("6. Saving the model...")

    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'spotseeker_nn_v1.joblib')
    joblib.dump(nn_model, model_path)
    print(f"   -> Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()

