# TemporaPark — Milestone 2: Model Comparison
# Generates Accuracy, F1-Score and Confusion Matrices for all three models

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    classification_report,
    confusion_matrix
)

# CONFIG

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
TARGET_COL       = "Available_Binary"
CATEGORICAL_COLS = ["Weather_Condition"]
LABEL_NAMES      = ["Available", "Unavailable"]
RANDOM_STATE     = 42
TEST_SIZE        = 0.20

# LOAD & PREPROCESS

def load_and_preprocess():
    df = pd.read_csv(PARKING_CSV_PATH)
    X  = df[FEATURE_COLS].copy()
    y  = df[TARGET_COL].copy()
    X  = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    return X, y

# TRAIN / TEST SPLIT

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    scaler         = StandardScaler()
    X_train_sc     = scaler.fit_transform(X_train)
    X_test_sc      = scaler.transform(X_test)
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test

# EVALUATE — confusion matrix + classification report for one model

def evaluate_model(model_name, y_test, y_pred):
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  Labels: {LABEL_NAMES}")
    print(cm)
    print(f"\n  Classification Report:")
    print(cr)

    return {"model": model_name, "accuracy": acc, "precision": prec, "f1": f1}

# MODELS

def run_decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE, criterion="gini")
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return evaluate_model("Decision Tree", y_test, y_pred)


def run_knn(X_train_sc, X_test_sc, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=4, metric="euclidean", weights="uniform")
    knn.fit(X_train_sc, y_train)
    y_pred = knn.predict(X_test_sc)
    return evaluate_model("KNN (k=4)", y_test, y_pred)


def run_neural_network(y_test, y_pred_from_anchal):
    """
    Plug in Anchal's Neural Network predictions here.
    y_pred_from_anchal should be a list or numpy array of 0s and 1s
    with the same length as y_test (1000 values).

    HOW TO GET IT FROM ANCHAL:
      1. She runs her MLPClassifier on the same test set
      2. She does: y_pred = model.predict(X_test)
      3. She sends you that array or saves it as nn_predictions.csv
      4. Load it here: y_pred_from_anchal = pd.read_csv("nn_predictions.csv")["y_pred"].values
    """
    if y_pred_from_anchal is None:
        print("\n  Neural Network — waiting for Anchal's predictions")
        return {"model": "Neural Network", "accuracy": None, "precision": None, "f1": None}

    return evaluate_model("Neural Network (MLP)", y_test, y_pred_from_anchal)

# COMPARISON TABLE

def print_comparison_table(results):
    print(f"\n{'='*55}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'F1-Score':>10}")
    print(f"  {'-'*55}")
    for r in results:
        if r["accuracy"] is None:
            print(f"  {r['model']:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"  {r['model']:<25} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% {r['f1']*100:>9.2f}%")
    print()

# MAIN

if __name__ == "__main__":

    X, y = load_and_preprocess()
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test = split_and_scale(X, y)

    results = []

    # Decision Tree
    results.append(run_decision_tree(X_train, X_test, y_train, y_test))

    # KNN
    results.append(run_knn(X_train_sc, X_test_sc, y_train, y_test))

    # Neural Network 
    nn_predictions = pd.read_csv("nn_predictions.csv")["y_pred"].values
    results.append(run_neural_network(y_test, nn_predictions))

    # Final comparison table
    print_comparison_table(results)