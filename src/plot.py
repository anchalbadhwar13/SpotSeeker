"""
visualizations_simple.py

Run:
    python visualizations_simple.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Paths
DATA_PATH = "synthetic_parking_dataset.csv"
MODEL_PATH = "models/decision_tree.pkl"
FEAT_PATH = "models/feature_cols.pkl"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. FEATURE IMPORTANCE
# =========================
def plot_feature_importance(model, features):
    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    plt.figure()
    plt.barh(df["Feature"], df["Importance"])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.show()


# =========================
# 2. OCCUPANCY VS HOUR
# =========================
def plot_occupancy_vs_hour(df):
    hourly = df.groupby("Hour")["Occupancy_Rate"].mean().sort_index()

    plt.figure()
    plt.plot(hourly.index, hourly.values, marker='o')
    plt.title("Occupancy vs Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average Occupancy")
    plt.grid()

    plt.savefig(os.path.join(OUTPUT_DIR, "occupancy_vs_hour.png"))
    plt.show()


# =========================
# 3. WEEKDAY VS WEEKEND
# =========================
def plot_weekday_vs_weekend(df):
    df["is_weekend"] = df["Day_of_Week"] >= 5

    weekday = df[df["is_weekend"] == False].groupby("Hour")["Occupancy_Rate"].mean()
    weekend = df[df["is_weekend"] == True].groupby("Hour")["Occupancy_Rate"].mean()

    plt.figure()
    plt.plot(weekday.index, weekday.values, label="Weekday")
    plt.plot(weekend.index, weekend.values, label="Weekend")
    plt.title("Weekday vs Weekend Occupancy")
    plt.xlabel("Hour")
    plt.ylabel("Occupancy")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(OUTPUT_DIR, "weekday_vs_weekend.png"))
    plt.show()


# =========================
# 4. HEATMAP
# =========================
def plot_heatmap(df):
    pivot = df.pivot_table(
        values="Occupancy_Rate",
        index="Day_of_Week",
        columns="Hour",
        aggfunc="mean"
    )

    plt.figure()
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Occupancy")
    plt.title("Occupancy Heatmap")
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")

    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap.png"))
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Loading model...")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        if os.path.exists(FEAT_PATH):
            features = joblib.load(FEAT_PATH)
        else:
            features = model.feature_names_in_
    else:
        print("Model not found. Skipping feature importance.")
        model = None

    print("Generating plots...")

    if model is not None:
        plot_feature_importance(model, features)

    plot_occupancy_vs_hour(df)
    plot_weekday_vs_weekend(df)
    plot_heatmap(df)

    print("Done! Check outputs/ folder.")