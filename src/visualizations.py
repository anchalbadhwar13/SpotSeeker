# import matplotlib.pyplot as plt
# import numpy as np

# # ── Confusion Matrices ──────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# dt_cm = np.array([[702, 53], [55, 190]])
# knn_cm = np.array([[677, 78], [182, 63]])
# labels = ['Available', 'Unavailable']

# for ax, cm, title in zip(axes, [dt_cm, knn_cm],
#                           ['Decision Tree (89.20%)', 'KNN (74.00%)']):
#     im = ax.imshow(cm, cmap='Blues')
#     ax.set_xticks([0,1]); ax.set_yticks([0,1])
#     ax.set_xticklabels(labels); ax.set_yticklabels(labels)
#     ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
#     ax.set_title(title)
#     for i in range(2):
#         for j in range(2):
#             ax.text(j, i, cm[i,j], ha='center', va='center',
#                    fontsize=16, color='white' if cm[i,j]>400 else 'black')

# plt.suptitle('SpotSeeker — Confusion Matrices', fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.savefig('confusion_matrices.png', dpi=150)
# plt.show()
# print('Saved confusion_matrices.png')

# # ── Feature Importance ──────────────────────────────────────────
# features = ['Hour', 'Parking_Duration_Min', 'Temperature_C', 'Day_of_Week', 'Month']
# importance = [0.766149, 0.070979, 0.060204, 0.049105, 0.025961]

# plt.figure(figsize=(9, 5))
# bars = plt.barh(features[::-1], importance[::-1], color='steelblue')
# plt.xlabel('Feature Importance')
# plt.title('Decision Tree — Top Feature Importances\nSpotSeeker Parking Prediction',
#           fontweight='bold')
# for bar, val in zip(bars, importance[::-1]):
#     plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
#              f'{val:.4f}', va='center', fontsize=10)
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=150)
# plt.show()
# print('Saved feature_importance.png')
"""
SpotSeeker — visualizations.py
Manya Asri | CMPT 310 Milestone 2
Generates all required visualizations and saves to outputs/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ── Setup ────────────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)

# ── Load processed data ──────────────────────────────────────────
print("Loading data...")
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()

df = pd.read_csv('synthetic_parking_dataset.csv')
print(f"Data loaded. Train: {X_train.shape}, Test: {X_test.shape}")

# ── Re-train models for predictions ─────────────────────────────
print("Training models for visualization...")
dt  = DecisionTreeClassifier(max_depth=10, random_state=42)
knn = KNeighborsClassifier(n_neighbors=4)
nn  = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)

dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
nn.fit(X_train, y_train)

dt_pred  = dt.predict(X_test)
knn_pred = knn.predict(X_test)
nn_pred  = nn.predict(X_test)

dt_acc  = (dt_pred  == y_test).mean() * 100
knn_acc = (knn_pred == y_test).mean() * 100
nn_acc  = (nn_pred  == y_test).mean() * 100

print(f"DT: {dt_acc:.1f}%  KNN: {knn_acc:.1f}%  NN: {nn_acc:.1f}%")


print("\nGenerating feature importance chart...")
feature_names = X_train.columns.tolist()
importances   = dt.feature_importances_
sorted_idx    = np.argsort(importances)

plt.figure(figsize=(10, 6))
bars = plt.barh([feature_names[i] for i in sorted_idx],
                importances[sorted_idx], color='steelblue')
plt.xlabel('Feature Importance')
plt.title('Decision Tree — Feature Importances\nSpotSeeker Parking Prediction',
          fontweight='bold')
for bar, val in zip(bars, importances[sorted_idx]):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved outputs/feature_importance.png")

print("Generating temporal patterns chart...")

df['Occ_Num'] = (df['Occupancy_Status'] == 'Unavailable').astype(int)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SpotSeeker — Temporal Occupancy Patterns', fontsize=15, fontweight='bold')

# Panel 1: Avg Occupancy Rate by Hour
# hourly = df.groupby('Hour')['Occupancy_Rate'].mean() * 100
# axes[0,0].plot(hourly.index, hourly.values, color='steelblue', linewidth=2, marker='o', markersize=4)
# axes[0,0].fill_between(hourly.index, hourly.values, alpha=0.2, color='steelblue')
# axes[0,0].set_title('Avg Occupancy Rate by Hour of Day')
# axes[0,0].set_xlabel('Hour'); axes[0,0].set_ylabel('Occupancy Rate (%)')
# axes[0,0].grid(True, alpha=0.3)

# # Panel 2: Occupancy by Day Name
# day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# daily = df.groupby('Day_Name')['Occupancy_Rate'].mean() * 100
# daily = daily.reindex(day_order)
# axes[0,1].bar(range(len(daily)), daily.values, color='coral', edgecolor='black')
# axes[0,1].set_xticks(range(len(daily)))
# axes[0,1].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
# axes[0,1].set_title('Avg Occupancy Rate by Day of Week')
# axes[0,1].set_xlabel('Day'); axes[0,1].set_ylabel('Occupancy Rate (%)')
# axes[0,1].grid(True, alpha=0.3, axis='y')

# # Panel 3: Occupancy Status counts by Hour
# for status, color in [('Empty','mediumseagreen'),('Moderate','gold'),
#                       ('Busy','coral'),('Full','crimson')]:
#     counts = df[df['Occupancy_Status']==status].groupby('Hour').size()
#     axes[1,0].plot(counts.index, counts.values, label=status, color=color, linewidth=2)
# axes[1,0].set_title('Occupancy Status Distribution by Hour')
# axes[1,0].set_xlabel('Hour'); axes[1,0].set_ylabel('Count')
# axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# # Panel 4: Occupancy by Month
# monthly = df.groupby('Month')['Occupancy_Rate'].mean() * 100
# axes[1,1].bar(monthly.index, monthly.values, color='mediumpurple', edgecolor='black')
# axes[1,1].set_title('Avg Occupancy Rate by Month')
# axes[1,1].set_xlabel('Month'); axes[1,1].set_ylabel('Occupancy Rate (%)')
# axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved outputs/temporal_patterns.png")

print("Generating occupancy heatmap...")
heatmap_data = df.groupby(['Hour', 'Day_Name'])['Occupancy_Rate'].mean() * 100
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heatmap_pivot = heatmap_data.unstack('Day_Name').reindex(columns=day_order)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_pivot, cmap='RdYlGn_r', annot=True, fmt='.0f',
            linewidths=0.3, cbar_kws={'label': 'Avg Occupancy Rate (%)'},
            vmin=20, vmax=80)
plt.title('SpotSeeker — Occupancy Heatmap (Hour × Day of Week)\nAverage Occupancy Rate %',
          fontsize=13, fontweight='bold')
plt.xlabel('Day of Week'); plt.ylabel('Hour of Day')
plt.tight_layout()
plt.savefig('outputs/occupancy_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved outputs/occupancy_heatmap.png")


print("Generating confusion matrices...")
labels     = ['Available', 'Unavailable']
preds      = [dt_pred, knn_pred, nn_pred]
titles     = [f'Decision Tree ({dt_acc:.1f}%)',
              f'KNN k=4 ({knn_acc:.1f}%)',
              f'Neural Network ({nn_acc:.1f}%)']
colors     = ['Blues', 'Oranges', 'Greens']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SpotSeeker — Confusion Matrices: All Models',
             fontsize=14, fontweight='bold')

for ax, pred, title, cmap in zip(axes, preds, titles, colors):
    cm = confusion_matrix(y_test, pred, labels=labels)
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels, rotation=15)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title, fontweight='bold')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha='center', va='center',
                    fontsize=18,
                    color='white' if cm[i,j] > thresh else 'black')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved outputs/confusion_matrices.png")


print("\n" + "="*50)
print("  ALL VISUALIZATIONS SAVED TO outputs/")
print("="*50)
print("  outputs/feature_importance.png")
print("  outputs/temporal_patterns.png")
print("  outputs/occupancy_heatmap.png")
print("  outputs/confusion_matrices.png")
print("="*50)
print(f"\n  Decision Tree  : {dt_acc:.2f}%")
print(f"  KNN (k=4)      : {knn_acc:.2f}%")
print(f"  Neural Network : {nn_acc:.2f}%")
print("="*50)
