"""
SpotSeeker CLI v2.0 — Predict parking spot availability by zone.
All predictions come from the trained model. Nothing is hardcoded.
Usage: python3 spotseeker_cli.py
"""

import sys
import joblib
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH    = "spotseeker_final_nn.joblib"
SCALER_PATH   = "spotseeker_scaler.joblib"
FEATURES_PATH = "spotseeker_feature_cols.joblib"

# ── Menu options (match exact strings in synthetic_parking_dataset.csv) ────────
DAYS = {
    "1": (0, "Monday"),    "2": (1, "Tuesday"),  "3": (2, "Wednesday"),
    "4": (3, "Thursday"),  "5": (4, "Friday"),   "6": (5, "Saturday"),
    "7": (6, "Sunday"),
}

MONTHS = {
    "1":  (1,  "January"),   "2":  (2,  "February"), "3":  (3,  "March"),
    "4":  (4,  "April"),     "5":  (5,  "May"),      "6":  (6,  "June"),
    "7":  (7,  "July"),      "8":  (8,  "August"),   "9":  (9,  "September"),
    "10": (10, "October"),   "11": (11, "November"), "12": (12, "December"),
}

WEATHER_OPTIONS = {
    "1": "Sunny",
    "2": "Cloudy",
    "3": "Overcast",
    "4": "Light Rain",
    "5": "Heavy Rain",
}

# Precipitation flag derived from weather — matches data_prep.py logic
WEATHER_PRECIPITATION = {
    "Sunny": 0, "Cloudy": 0, "Overcast": 0,
    "Light Rain": 1, "Heavy Rain": 1,
}

ZONES = {
    "1": "Zone_A_Downtown",
    "2": "Zone_B_University",
    "3": "Zone_C_Shopping",
    "4": "Zone_D_Residential",
    "5": "Zone_E_Airport",
}

ZONE_CAPACITIES = {
    "Zone_A_Downtown":    200,
    "Zone_B_University":  150,
    "Zone_C_Shopping":    300,
    "Zone_D_Residential": 100,
    "Zone_E_Airport":     250,
}

# ANSI colours
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED  = "\033[91m"
CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET = "\033[0m"

# ── UI Helpers ─────────────────────────────────────────────────────────────────
def banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════╗
║        SpotSeeker  CLI  v2.0         ║
║   AI-powered parking availability    ║
╚══════════════════════════════════════╝{RESET}
""")

def prompt_menu(title, options, label="Choice"):
    if title:
        print(f"{BOLD}{title}{RESET}")
    for k, v in options.items():
        display = v if isinstance(v, str) else v[1]
        print(f"  {k:>2}) {display}")
    while True:
        choice = input(f"{label}: ").strip()
        if choice in options:
            return choice, options[choice]
        print(f"  ⚠  Invalid — enter a number from the list.")

def prompt_hour():
    print(f"\n{BOLD}Enter hour (0–23, 24-hour format){RESET}")
    print("  e.g. 9 = 9 AM, 14 = 2 PM, 17 = 5 PM")
    while True:
        raw = input("Hour: ").strip()
        if raw.isdigit() and 0 <= int(raw) <= 23:
            return int(raw)
        print("  ⚠  Enter a whole number between 0 and 23.")

def prompt_temperature():
    print(f"\n{BOLD}Outside temperature (°C){RESET}")
    print("  e.g. -5 = cold, 0 = freezing, 12 = mild, 25 = warm")
    while True:
        raw = input("Temp °C: ").strip()
        try:
            t = float(raw)
            if -40 <= t <= 50:
                return t
        except ValueError:
            pass
        print("  ⚠  Enter a number between -40 and 50.")

def prompt_duration():
    print(f"\n{BOLD}Expected parking duration (minutes){RESET}")
    print("  e.g. 30 = half hour, 60 = 1 hr, 120 = 2 hrs, 480 = all day")
    while True:
        raw = input("Duration (min): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 1440:
            return int(raw)
        print("  ⚠  Enter a number between 1 and 1440.")

def prompt_special_event():
    print(f"\n{BOLD}Is there a special event nearby? (concert, game, festival…){RESET}")
    print("  1) No\n  2) Yes")
    while True:
        raw = input("Choice: ").strip()
        if raw == "1": return 0
        if raw == "2": return 1
        print("  ⚠  Enter 1 or 2.")

def hour_label(h):
    ampm = "AM" if h < 12 else "PM"
    h12  = 12 if h == 0 else (h if h <= 12 else h - 12)
    return f"{h12}:00 {ampm}"

def status_label(prob_available):
    pct_full = 1 - prob_available
    if pct_full < 0.30:
        return f"🟢  Open — plenty of spots  ({prob_available*100:.0f}% chance available)", GREEN
    elif pct_full < 0.60:
        return f"🟡  Moderate — some spots free  ({prob_available*100:.0f}% chance available)", YELLOW
    elif pct_full < 0.85:
        return f"🟠  Busy — limited availability  ({prob_available*100:.0f}% chance available)", YELLOW
    else:
        return f"🔴  Full — very likely no spots  ({prob_available*100:.0f}% chance available)", RED

# ── Feature builder ────────────────────────────────────────────────────────────
def build_feature_row(hour, day_of_week, month, weather, zone_key,
                      temp_c, duration_min, special_event, feature_cols):
    """
    Builds one unscaled feature row that exactly matches the column
    order and one-hot structure produced by data_prep.py.

    Strategy: start every column at 0, then fill in numeric values
    and flip the correct one-hot columns to 1.
    """
    row = {col: 0 for col in feature_cols}

    # ── Numeric features ───────────────────────────────────────────────
    row['Hour']                 = hour
    row['Day_of_Week']          = day_of_week
    row['Month']                = month
    row['Precipitation']        = WEATHER_PRECIPITATION[weather]
    row['Special_Event']        = special_event
    row['Temperature_C']        = temp_c
    row['Parking_Duration_Min'] = duration_min

    # ── One-hot: Weather_Condition ─────────────────────────────────────
    # data_prep.py creates columns like: Weather_Condition_Sunny, etc.
    weather_col = f"Weather_Condition_{weather}"
    if weather_col in row:
        row[weather_col] = 1
    else:
        print(f"  ⚠  Warning: column '{weather_col}' not found in model features.")

    # ── One-hot: Parking_Zone_ID ───────────────────────────────────────
    zone_col = f"Parking_Zone_ID_{zone_key}"
    if zone_col in row:
        row[zone_col] = 1
    else:
        print(f"  ⚠  Warning: column '{zone_col}' not found in model features.")

    # Return DataFrame with columns in exact training order
    return pd.DataFrame([row])[feature_cols]

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    banner()

    # Load all three model artifacts
    try:
        model        = joblib.load(MODEL_PATH)
        scaler       = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
    except FileNotFoundError as e:
        print(f"{RED}❌ Missing file: {e}{RESET}")
        print("Make sure you have run:")
        print("  1. python3 data_prep.py")
        print("  2. python3 train_nn.py")
        sys.exit(1)

    print(f"✅ Model loaded  |  {len(feature_cols)} features  |  Classes: {list(model.classes_)}\n")

    while True:
        # ── Collect all inputs from the user ──────────────────────────
        print(f"{BOLD}─── Step 1 of 6: Day of week ──────────────────{RESET}")
        _, day_val = prompt_menu("", DAYS, label="Day")
        day_num, day_name = day_val

        hour = prompt_hour()

        print(f"\n{BOLD}─── Step 2 of 6: Month ────────────────────────{RESET}")
        _, month_val = prompt_menu("", MONTHS, label="Month")
        month_num, month_name = month_val

        print(f"\n{BOLD}─── Step 3 of 6: Weather ──────────────────────{RESET}")
        _, weather = prompt_menu("", WEATHER_OPTIONS, label="Weather")

        temp = prompt_temperature()

        print(f"\n{BOLD}─── Step 4 of 6: Duration ─────────────────────{RESET}")
        duration = prompt_duration()

        print(f"\n{BOLD}─── Step 5 of 6: Special event ────────────────{RESET}")
        special_event = prompt_special_event()

        print(f"\n{BOLD}─── Step 6 of 6: Zone ─────────────────────────{RESET}")
        zone_choice, _ = prompt_menu("", ZONES, label="Zone")
        zone_key  = ZONES[zone_choice]
        capacity  = ZONE_CAPACITIES[zone_key]

        # ── Build feature row → scale → predict ───────────────────────
        X_raw = build_feature_row(
            hour, day_num, month_num, weather, zone_key,
            temp, duration, special_event, feature_cols
        )

        X_scaled = scaler.transform(X_raw)          # ← NEVER .fit() here
        proba    = model.predict_proba(X_scaled)[0]
        classes  = list(model.classes_)

        # Find P(Available) — handles both string and numeric class labels
        if "Available" in classes:
            p_avail = proba[classes.index("Available")]
        elif 1 in classes:
            p_avail = proba[classes.index(1)]
        else:
            p_avail = proba[0]   # fallback

        est_free = round(capacity * p_avail)

        # ── Display result ─────────────────────────────────────────────
        label, colour = status_label(p_avail)
        zone_display  = zone_key.replace("_", " ")

        print(f"""
{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
  {BOLD}Prediction Summary{RESET}
  Zone      : {zone_display}
  When      : {day_name}, {month_name}, {hour_label(hour)}
  Weather   : {weather}  ({temp}°C)
  Duration  : {duration} min
  Spec. Event: {"Yes" if special_event else "No"}
{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
  {colour}{BOLD}{label}{RESET}
  Est. free spots : {colour}{BOLD}{est_free}{RESET} / {capacity}
{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
""")

        again = input("Check another spot? (y/n): ").strip().lower()
        if again != "y":
            print(f"\n{CYAN}Thanks for using SpotSeeker! 🚗{RESET}\n")
            break
        print()

if __name__ == "__main__":
    main()