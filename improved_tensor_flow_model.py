from base_model import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImprovedTensorFlowModel(BaseModel):
    def __init__(self, train=False):
        super().__init__()
        if train:
            self.train_model()
        else:
            try:
                self.load_model()
                print("Improved model loaded!")
            except:
                self.train_model()
                print("Improved model trained!")

    def load_model(self):
        self.model = tf.keras.models.load_model('improved_character_model.keras')
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

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        # Build improved model with deeper architecture
        self.model = tf.keras.Sequential([
            # First convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(images.shape[1], images.shape[2], 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Second convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Third convolutional block
            tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Fully connected layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.label_to_index), activation='softmax')
        ])

        # Use a learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(optimizer=optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Add early stopping and model checkpoint callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'improved_character_model_checkpoint.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

        # Train model with data augmentation
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=50,  # Increased from 10 to 50
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint]
        )

        # Save model
        self.model.save('improved_character_model.keras')
        print("Improved model saved!")

    def scan_img(self, img, return_confidence=False, return_top_k=False, k=3):
        prediction = self.model.predict(img)
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
    model = ImprovedTensorFlowModel(train=True)
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)
