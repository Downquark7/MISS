from base_model import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


class TensorFlowModel(BaseModel):
    """TensorFlow-based CNN model for character recognition.

    This model uses a Convolutional Neural Network implemented in TensorFlow
    for character recognition.
    """

    def __init__(self, train=False, model_path='character_model.keras'):
        """Initialize the TensorFlow model.

        Args:
            train (bool, optional): Whether to train the model. Defaults to False.
            model_path (str, optional): Path to the saved model. Defaults to 'character_model.keras'.
        """
        super().__init__()
        self.model_path = model_path

        if train:
            self.train_model()
        else:
            try:
                self.load_model()
                print("Model loaded!")
            except (IOError, ImportError, ValueError) as e:
                print(f"Error loading model: {e}")
                print("Training new model instead...")
                self.train_model()
                print("Model trained!")

    def load_model(self, label_mappings_path='label_mappings.npy', labels_file='labels.csv'):
        """Load the trained model and label mappings.

        Args:
            label_mappings_path (str, optional): Path to the label mappings file. Defaults to 'label_mappings.npy'.
            labels_file (str, optional): Path to the labels CSV file. Defaults to 'labels.csv'.

        Raises:
            IOError: If the model or label mappings file cannot be found.
            ImportError: If there's an issue with the model format.
            ValueError: If the model or label mappings are invalid.
        """
        self.model = tf.keras.models.load_model(self.model_path)
        self.index_to_label = np.load(label_mappings_path, allow_pickle=True).item()
        self.labels_df = pd.read_csv(labels_file)
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(self.labels_df['label']))}

    def train_model(self, labels_file='labels.csv', image_folder='labeled_characters_binary',
                  label_mappings_path='label_mappings.npy', test_size=0.2, random_state=42,
                  epochs=10):
        """Train the TensorFlow CNN model.

        Args:
            labels_file (str, optional): Path to the CSV file containing labels. Defaults to 'labels.csv'.
            image_folder (str, optional): Path to the folder containing images. Defaults to 'labeled_characters_binary'.
            label_mappings_path (str, optional): Path to save label mappings. Defaults to 'label_mappings.npy'.
            test_size (float, optional): Proportion of data to use for validation. Defaults to 0.2.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            epochs (int, optional): Number of training epochs. Defaults to 10.
        """
        # Load and preprocess data
        self.labels_df = pd.read_csv(labels_file)

        # Create label mappings
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(self.labels_df['label']))}
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        # Save label mappings for later use
        np.save(label_mappings_path, self.index_to_label)

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
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=random_state)

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
        self.history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

        # Save model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}!")

    def scan_img(self, img, return_confidence=False, return_top_k=False, k=3, verbose=True):
        """Scan an image and predict the character.

        Args:
            img (numpy.ndarray): The image to scan.
            return_confidence (bool, optional): Whether to return confidence score. Defaults to False.
            return_top_k (bool, optional): Whether to return top k predictions. Defaults to False.
            k (int, optional): Number of top predictions to return if return_top_k is True. Defaults to 3.
            verbose (bool, optional): Whether to print prediction details. Defaults to True.

        Returns:
            str, tuple, or list: 
                - If return_top_k is True, returns a list of (character, confidence) tuples for top k predictions.
                - If return_confidence is True, returns a tuple (character, confidence).
                - Otherwise, returns the predicted character as a string.
        """
        prediction = self.model.predict(img, verbose=0)  # Suppress TensorFlow output
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        top_5_characters = [(self.index_to_label[idx], prediction[0][idx]) for idx in top_5_indices]

        if verbose:
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
    """Main entry point for testing the TensorFlow model."""
    model = TensorFlowModel(train=False)
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)
