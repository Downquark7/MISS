from base_model import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


class TensorFlowModel(BaseModel):
    def __init__(self, train=False):
        super().__init__()
        if train:
            self.train_model()
        else:
            try:
                self.load_model()
                print("Model loaded!")
            except:
                self.train_model()
                print("Model trained!")

    def load_model(self):
        self.model = tf.keras.models.load_model('character_model.keras')
        self.index_to_label = np.load('label_mappings.npy', allow_pickle=True).item()
        self.labels_df = pd.read_csv('labels.csv')
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(self.labels_df['label']))}

    def train_model(self):
        # Load and preprocess data
        self.labels_df = pd.read_csv('labels.csv')
        image_folder = 'labeled_characters_binary'

        # Create label mappings
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(self.labels_df['label']))}
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        # Save label mappings for later use
        np.save('label_mappings.npy', self.index_to_label)

        # Load images and labels
        images = []
        labels = []
        for filename, label in zip(self.labels_df['filename'], self.labels_df['label']):
            img_path = os.path.join(image_folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale')
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img = self.pad_image(img_array)
            images.append(img / 255.0)  # Normalize
            labels.append(self.label_to_index[label])

        images = np.array(images)
        labels = np.array(labels)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Build model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.label_to_index), activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Train model
        self.history = self.model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

        # Save model
        self.model.save('character_model.keras')
        print("Model saved!")

    def scan_img(self, img):
        prediction = self.model.predict(img)
        predicted_index = np.argmax(prediction)
        certainty = prediction[0][predicted_index]
        character = self.index_to_label[predicted_index]
        print(f"Predicted character: {character}, Certainty: {certainty:.2f}")
        return character


if __name__ == "__main__":
    model = TensorFlowModel(train=True)
    # model.train_model()
    # model.load_model()
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)
