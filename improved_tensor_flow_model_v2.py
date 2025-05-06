from base_model import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


class ImprovedTensorFlowModelV2(BaseModel):
    def __init__(self, train=False):
        super().__init__()
        if train:
            self.train_model()
        else:
            try:
                self.load_model()
                print("Improved model V2 loaded!")
            except:
                self.train_model()
                print("Improved model V2 trained!")

    def load_model(self):
        self.model = tf.keras.models.load_model('improved_character_model_v2.keras')
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

        # Reshape images to have 4 dimensions (batch_size, height, width, channels)
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Build improved model - similar to original but with some enhancements
        self.model = tf.keras.Sequential([
            # First convolutional block - same as original but with more filters
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(images.shape[1], images.shape[2], 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Second convolutional block - same as original but with more filters
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Additional convolutional layer
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

            # Fully connected layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),  # Increased from 64 to 128
            tf.keras.layers.Dropout(0.3),  # Add dropout for regularization
            tf.keras.layers.Dense(len(self.label_to_index), activation='softmax')
        ])

        # Compile model with the same settings as original
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=20,  # Increased from 10 to 20
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )

        # Save model
        self.model.save('improved_character_model_v2.keras')
        print("Improved model V2 saved!")

    def scan_img(self, img, return_confidence=False, return_top_k=False, k=3):
        # Use __call__ instead of predict to avoid TensorFlow compatibility issues
        prediction = self.model(img, training=False).numpy()
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        top_5_characters = [(self.index_to_label[idx], prediction[0][idx]) for idx in top_5_indices]
        for character, certainty in top_5_characters:
            print(f"Character: {character}, Certainty: {certainty:.2f}")
        predicted_index = top_5_indices[0]
        character = self.index_to_label[predicted_index]
        confidence = prediction[0][predicted_index]

        if return_top_k:
            # Return top k predictions with their confidence scores
            top_k_indices = np.argsort(prediction[0])[-k:][::-1]
            top_k_predictions = [(self.index_to_label[idx], prediction[0][idx]) for idx in top_k_indices]
            return top_k_predictions
        elif return_confidence:
            return character, confidence
        else:
            return character


if __name__ == "__main__":
    model = ImprovedTensorFlowModelV2(train=True)
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)
