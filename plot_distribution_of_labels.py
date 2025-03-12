import pandas as pd

# Load and preprocess data
labels_df = pd.read_csv('labels.csv')
image_folder = 'labeled_characters_binary'

# Plot distribution of labels
import matplotlib.pyplot as plt

label_counts = labels_df['label'].value_counts()
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.title('Distribution of Labels')
plt.xlabel('Labels')
plt.ylabel('Number of Images')
plt.show()