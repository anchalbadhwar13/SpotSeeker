import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

print("Loading data from processed_data/...")
X_train = pd.read_csv('processed_data/X_train.csv')
X_test  = pd.read_csv('processed_data/X_test.csv')
y_train = pd.read_csv('processed_data/y_train.csv').squeeze()
y_test  = pd.read_csv('processed_data/y_test.csv').squeeze()

df = pd.read_csv('synthetic_parking_dataset.csv')
df.columns = [c.lower().replace(' ', '_') for c in df.columns]

print(f"✅ Data loaded. Train: {X_train.shape}, Test: {X_test.shape}")

print("Training models for visualization...")
dt  = DecisionTreeClassifier(max_depth=10, random_state=42)
knn = KNeighborsClassifier(n_neighbors=4)
nn  = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=42)

dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
nn.fit(X_train, y_train)

dt_pred  = dt.predict(X_test)
knn_pred = knn.predict(X_test)
nn_pred  = nn.predict(X_test)

dt_acc  = (dt_pred  == y_test).mean() * 100
knn_acc = (knn_pred == y_test).mean() * 100
nn_acc  = (nn_pred  == y_test).mean() * 100

print("\nGenerating feature importance chart...")
feature_names = X_train.columns.tolist()
importances   = dt.feature_importances_
sorted_idx    = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color='steelblue')
plt.title('Feature Importance (Decision Tree)')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
plt.close()

print("Generating temporal patterns...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

hourly = df.groupby('hour')['occupancy_status'].mean() * 100
axes[0].plot(hourly.index, hourly.values, marker='o', color='steelblue')
axes[0].set_title('Avg Occupancy by Hour')

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
daily = df.groupby('day_name')['occupancy_rate'].mean() * 100
daily = daily.reindex(day_order)
axes[1].bar(daily.index, daily.values, color='coral')
axes[1].set_title('Avg Occupancy by Day')

plt.tight_layout()
plt.savefig('outputs/temporal_patterns.png')
plt.close()

print("Generating confusion matrices...")
# Instead of hardcoding 'Available', we detect the classes actually in the data
unique_labels = sorted(np.unique(y_test)) 
preds  = [dt_pred, knn_pred, nn_pred]
titles = [f'DT ({dt_acc:.1f}%)', f'KNN ({knn_acc:.1f}%)', f'NN ({nn_acc:.1f}%)']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, pred, title in zip(axes, preds, titles):
    cm = confusion_matrix(y_test, pred, labels=unique_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=unique_labels, yticklabels=unique_labels)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png')
plt.close()

print("\n" + "="*50)
print("  ALL VISUALIZATIONS SAVED TO outputs/")
print("="*50)