import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("   MILESTONE 2 — MODEL COMPARISON REPORT")
print("=" * 50)

models     = ['Decision Tree', 'KNN (k=5)', 'KNN (k=4, tuned)', 'Neural Network']
accuracies = [89.20, 74.00, 76.20, 85.10]

print(f"\n{'Model':<25} {'Accuracy':>10}")
print("-" * 37)
for m, a in zip(models, accuracies):
    print(f"{m:<25} {a:>9.2f}%")
print("-" * 37)
print(f"Best Model: Decision Tree     89.20%")

colors = ['steelblue', 'coral', 'lightcoral', 'mediumseagreen']
plt.figure(figsize=(9, 5))
bars = plt.bar(models, accuracies, color=colors, edgecolor='black')
plt.ylim(60, 100)
plt.ylabel('Accuracy (%)')
plt.title('SpotSeeker — Milestone 2 Model Comparison', fontweight='bold')
for bar, val in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('milestone2_comparison.png', dpi=150)
plt.show()
print("\nSaved milestone2_comparison.png")

