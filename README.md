# 🚗 SpotSeeker: AI Parking Spot Prediction

**SpotSeeker** is a machine learning-based system designed to tackle urban parking congestion. By utilizing a Multi-Layer Perceptron (MLP) Neural Network, the system predicts parking occupancy levels across various city zones with high precision, helping drivers find available spots and reducing traffic search time.

## 👥 The Team
* **Veer** – System Integration, Persistence, & CLI Development
* **Manya** – Problem Definition & Data Strategy
* **Anchal** – Model Architecture & Training
* **Ira** – Data Visualization & Reporting

---

## 📈 Performance Highlights
* **Final Accuracy:** 84.87% (Neural Network MLP)
* **Dataset:** 35,000 Synthetic Rows (Augmented for optimal generalization)
* **Target Classes:** 4-level occupancy (Empty, Moderate, Busy, Full)
* **Key Metric:** 91% Recall for "Available" spots, ensuring high reliability for users.

---

## 🛠️ Technical Features
* **Cyclical Feature Engineering:** Implemented Sine/Cosine transformations for time-based data to preserve temporal continuity.
* **Model Persistence:** Utilized `joblib` for serialized model and scaler synchronization, ensuring consistent inference.
* **Class Balancing:** Applied automated sample weighting to handle class imbalance in "Unavailable" parking scenarios.
* **Interactive CLI:** A terminal-based interface for real-time occupancy queries.

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anchalbadhwar13/SpotSeeker.git
   cd SpotSeeker
   ```

2. **Set up the environment:**
   Ensure you have Python 3.x installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 How to Run

Follow these steps in order to replicate our results:

1. **Data Preprocessing:**
   Cleans the raw data and performs feature scaling.
   ```bash
   python3 src/data_prep.py
   ```

2. **Model Training:**
   Trains the Neural Network and saves the model to the `models/` directory.
   ```bash
   python3 src/train_nn.py
   ```

3. **Interactive Inference (Demo):**
   Launch the CLI to test the model with custom inputs.
   ```bash
   python3 src/spotseeker_cli.py
   ```

---

## 📂 Project Structure
```text
SpotSeeker/
├── synthetic_parking_data.csv        
├── requirements.txt        # Environment dependencies
├── README.md               # Project documentation
├── models/                 # Serialized .joblib models & scalers
├── processed_data/         # Training/Testing CSV splits
└── src/                    # Source code
    ├── DataGen.py
    ├── data_prep.py
    ├── train_nn.py
    └── spotseeker_cli.py
```

---

## ⚖️ License
Developed for **CMPT 310 - Introduction to Artificial Intelligence (Spring 2026)**.