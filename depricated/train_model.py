import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from training_data_test import load_and_pad_image

# Load and preprocess data
labels_df = pd.read_csv('../labels.csv')
image_folder = 'labeled_characters_binary'


# Create label mappings
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels_df['label']))}
index_to_label = {v: k for k, v in label_to_index.items()}

# Save label mappings for later use
np.save('../label_mappings.npy', index_to_label)

# Load images and labels
images = []
labels = []
for filename, label in zip(labels_df['filename'], labels_df['label']):
    img_path = os.path.join(image_folder, filename)
    img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = load_and_pad_image(img_path, target_size=(28, 28))
    images.append(img / 255.0)  # Normalize
    labels.append(label_to_index[label])

images = np.array(images)
labels = np.array(labels)

# Split data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(images.shape[1], images.shape[2], 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_to_index), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save model
model.save('character_model.keras')
print("Model saved!")