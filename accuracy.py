import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
results_path = "cell_counting/features/evaluation_results.csv"
results_df = pd.read_csv(results_path, index_col=0)

# Plotting accuracy (within ±5% of the actual count)
plt.figure(figsize=(10, 6))
accuracy_values = results_df.loc['Accuracy_within_5%']
accuracy_values.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])

plt.xlabel('Dataset')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.title('Accuracy within ±5% of Actual Count for Training, Validation, and Test Sets')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding labels on top of each bar
for index, value in enumerate(accuracy_values):
    plt.text(index, value + 1, f'{value:.2f}%', ha='center')

plt.tight_layout()
plt.savefig("cell_counting/models/accuracy_plot.png")
plt.show()

# Plotting R² Scores
plt.figure(figsize=(10, 6))
r2_values = results_df.loc['R²']
r2_values.plot(kind='bar', color=['dodgerblue', 'limegreen', 'coral'])

plt.xlabel('Dataset')
plt.ylabel('R² Score')
plt.ylim(-1, 1)
plt.title('R² Score for Training, Validation, and Test Sets')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding labels on top of each bar
for index, value in enumerate(r2_values):
    plt.text(index, value + 0.05, f'{value:.2f}', ha='center')

plt.tight_layout()
plt.savefig("cell_counting/models/r2_plot.png")
plt.show()
