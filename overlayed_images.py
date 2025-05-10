from matplotlib import pyplot as plt

from base_model import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from base_model import BaseModel

model = BaseModel()

labels_df = pd.read_csv('labels.csv')
image_folder = 'labeled_characters_binary'

# Create label mappings
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels_df['label']))}
index_to_label = {v: k for k, v in label_to_index.items()}

# Save label mappings for later use
np.save('label_mappings.npy', index_to_label)

# Load images and labels
images = []
labels = []
for filename, label in zip(labels_df['filename'], labels_df['label']):
    img_path = os.path.join(image_folder, filename)
    img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img = model.pad_image(img_array)
    images.append(img / 255.0)  # Normalize
    labels.append(label_to_index[label])

images = np.array(images)
labels = np.array(labels)


def overlay_images_by_label(label):
    label_index = label_to_index[label]
    indices = np.where(labels == label_index)
    selected_images = images[indices]
    stacked_image = np.mean(selected_images, axis=0)
    return stacked_image

for label in label_to_index:
    stacked_image = overlay_images_by_label(label)
    if stacked_image.shape == ():
        continue
    log_image = np.log(stacked_image + 1e-8)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(stacked_image, cmap='gray')
    ax1.set_title('Original')
    ax2.imshow(log_image, cmap='gray')
    ax2.set_title('Log')
    plt.show()