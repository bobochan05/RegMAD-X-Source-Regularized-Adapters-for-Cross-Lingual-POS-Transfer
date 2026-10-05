import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the summary we just created
df = pd.read_csv("data_summary.csv")

# Filter for 'train' split to compare the training capacity for adapters
train_df = df[df['Split'] == 'train'].sort_values(by='Tokens', ascending=False)

# Set the visual style
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Tokens (Bar Chart)
sns.barplot(x="Language", y="Tokens", data=train_df, palette="viridis", ax=ax1)
ax1.set_title('Training Data Distribution (Tokens vs. Sentences)', fontsize=16)
ax1.set_ylabel('Total Tokens', fontsize=12, color='b')

# Create a second y-axis for Sentences (Line Plot)
ax2 = ax1.twinx()
sns.lineplot(x="Language", y="Sentences", data=train_df, marker='o', color='red', linewidth=2.5, ax=ax2)
ax2.set_ylabel('Total Sentences', fontsize=12, color='r')

# Add values on top of bars
for i, v in enumerate(train_df['Tokens']):
    ax1.text(i, v + 5000, f'{v:,}', color='black', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('dataset_distribution.png')
print("✅ Visualization saved as 'dataset_distribution.png'")
