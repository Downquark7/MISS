import os
import re
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def pad_image(img_array, target_size):
    # Get the original dimensions
    original_height, original_width, _ = img_array.shape
    target_width, target_height = target_size

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Resize the image while maintaining the aspect ratio
    if aspect_ratio > target_aspect_ratio:
        # Fit to width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Fit to height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_img = tf.image.resize(img_array, [new_height, new_width])

    # Create a black background image of the target size
    padded_img = tf.image.pad_to_bounding_box(
        resized_img,
        offset_height=(target_height - new_height) // 2,
        offset_width=(target_width - new_width) // 2,
        target_height=target_height,
        target_width=target_width
    )

    return padded_img


if __name__ == "__main__":
    # Model and label loading
    model = tf.keras.models.load_model('character_model.keras')
    index_to_label = np.load('label_mappings.npy', allow_pickle=True).item()
    labels_df = pd.read_csv('labels.csv')
    label_to_index = {label: idx for idx, label in enumerate(np.unique(labels_df['label']))}

def load_and_pad_image(img_path, target_size):
    # Load the image
    img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    return pad_image(img_array, target_size)


def check_image(img_path):
    padded_img = load_and_pad_image(
        img_path,
        target_size=(28, 28))
    img = np.array(padded_img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    character = index_to_label[predicted_index]
    actual_label = labels_df.loc[labels_df['filename'] == os.path.basename(img_path), 'label'].values[0]
    if character != actual_label:
        print(f"Incorrect prediction for {img_path}")
        print(f"Predicted: {character}, Actual: {actual_label}")
        plt.title(f"Predicted: {character}, Actual: {actual_label}")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(padded_img), cmap='gray')
        plt.show()


def numerical_sort(file):
    match = re.search(r'(\d+)', file)
    return int(match.group(1)) if match else float('inf')

if __name__ == "__main__":
    folder_path = 'labeled_characters_binary'
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]
    file_paths = sorted(file_paths, key = numerical_sort)

    for img_path in file_paths:
        check_image(img_path)