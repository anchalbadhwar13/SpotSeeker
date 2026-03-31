import matplotlib.pyplot as plt
import numpy as np

# ── Confusion Matrices ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
dt_cm = np.array([[702, 53], [55, 190]])
knn_cm = np.array([[677, 78], [182, 63]])
labels = ['Available', 'Unavailable']

for ax, cm, title in zip(axes, [dt_cm, knn_cm],
                          ['Decision Tree (89.20%)', 'KNN (74.00%)']):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha='center', va='center',
                   fontsize=16, color='white' if cm[i,j]>400 else 'black')

plt.suptitle('SpotSeeker — Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print('Saved confusion_matrices.png')

# ── Feature Importance ──────────────────────────────────────────
features = ['Hour', 'Parking_Duration_Min', 'Temperature_C', 'Day_of_Week', 'Month']
importance = [0.766149, 0.070979, 0.060204, 0.049105, 0.025961]

plt.figure(figsize=(9, 5))
bars = plt.barh(features[::-1], importance[::-1], color='steelblue')
plt.xlabel('Feature Importance')
plt.title('Decision Tree — Top Feature Importances\nSpotSeeker Parking Prediction',
          fontweight='bold')
for bar, val in zip(bars, importance[::-1]):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print('Saved feature_importance.png')