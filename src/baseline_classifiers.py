# Models: Decision Tree & KNN Classifier

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


USE_DUMMY_DATA   = False         
PARKING_CSV_PATH = "synthetic_parking_dataset.csv" 

FEATURE_COLS = [
    "Hour",
    "Day_of_Week",
    "Month",
    "Weather_Condition",
    "Precipitation",
    "Special_Event",
    "Temperature_C",
    "Parking_Duration_Min"
]
TARGET_COL = "Available_Binary"


CATEGORICAL_COLS = ["Weather_Condition"]

RANDOM_STATE = 42
TEST_SIZE    = 0.20   



def load_data():
    
    df = pd.read_csv(PARKING_CSV_PATH)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"These columns are missing from the CSV: {missing}\n"
            f"Expected: {FEATURE_COLS + [TARGET_COL]}\n"
            f"Found:    {list(df.columns)}"
        )

    return df


def preprocess(df):
   
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   
    X_test_scaled  = scaler.transform(X_test)


    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


LABEL_NAMES = ["Available", "Unavailable"]

def evaluate(model_name, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0)

    
    print(f"  {model_name}")
    
    print(f"  Accuracy: {acc * 100:.2f}%\n")
    print("  Confusion Matrix:")
    print("  (rows=actual, cols=predicted)")
    print(f"  Labels: {LABEL_NAMES}")
    print(cm)
    print()
    print("  Classification Report:")
    print(cr)
    return acc


def train_decision_tree(X_train, X_test, y_train, y_test):
    """
    Decision Tree classifier.
    max_depth=10 limits overfitting — the tree can't memorise training data
    by growing indefinitely. You can tune this value.
    """
    dt = DecisionTreeClassifier(
        max_depth=10,
        random_state=RANDOM_STATE,
        criterion="gini"    
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = evaluate("Decision Tree Classifier", y_test, y_pred)

    # Bonus
    feature_names = X_train.columns if hasattr(X_train, "columns") else range(X_train.shape[1])
    importances = pd.Series(dt.feature_importances_, index=feature_names)
    print("  Top feature importances:")
    print(importances.sort_values(ascending=False).head(5).to_string())
    print()

    return dt, acc


def train_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    K-Nearest Neighbours classifier.
    Uses SCALED features (StandardScaler applied above).
    k=5 means each prediction is decided by voting among the 5 nearest neighbours.
    """
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",   
        weights="uniform"     
    )
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = evaluate("KNN Classifier (k=5)", y_test, y_pred)
    return knn, acc


def tune_knn(X_train_scaled, X_test_scaled, y_train, y_test, k_range=range(1, 21)):
    """
    Tests multiple values of k and reports which gives the highest accuracy.
    This is basic grid search — scikit-learn's GridSearchCV can do this more
    formally, but this version keeps it transparent.
    """
    
    print("  KNN Hyperparameter Tuning — searching best k")
    

    results = {}
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, m.predict(X_test_scaled))
        results[k] = acc

    best_k   = max(results, key=results.get)
    best_acc = results[best_k]
    print(f"\n  Best k = {best_k}  (accuracy: {best_acc*100:.2f}%)\n")
    return best_k

def print_summary(dt_acc, knn_acc):
    
    print("  BASELINE SUMMARY")
    
    print(f"  Decision Tree accuracy : {dt_acc  * 100:.2f}%")
    print(f"  KNN accuracy           : {knn_acc * 100:.2f}%")
    print()

if __name__ == "__main__":

    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(X, y)
    dt_model, dt_acc = train_decision_tree(X_train, X_test, y_train, y_test)
    knn_model, knn_acc = train_knn(X_train_sc, X_test_sc, y_train, y_test)
    best_k = tune_knn(X_train_sc, X_test_sc, y_train, y_test, k_range=range(1, 16))
    print_summary(dt_acc, knn_acc)

