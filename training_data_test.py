import os
import re
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Model and label loading
model = tf.keras.models.load_model('character_model.keras')
index_to_label = np.load('label_mappings.npy', allow_pickle=True).item()
labels_df = pd.read_csv('labels.csv')
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels_df['label']))}


def check_image(img_path):
        img_read = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
        img = tf.keras.preprocessing.image.img_to_array(img_read)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)
        character = index_to_label[predicted_index]
        actual_label = labels_df.loc[labels_df['filename'] == os.path.basename(img_path), 'label'].values[0]
        if character != actual_label:
            print(f"Incorrect prediction for {img_path}")
            print(f"Predicted: {character}, Actual: {actual_label}")
            plt.title(f"Predicted: {character}, Actual: {actual_label}")
            plt.imshow(img_read, cmap='gray')
            plt.show()


def numerical_sort(file):
    match = re.search(r'(\d+)', file)
    return int(match.group(1)) if match else float('inf')


folder_path = 'labeled_characters_binary'
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]
file_paths = sorted(file_paths, key = numerical_sort)

for img_path in file_paths:
    check_image(img_path)