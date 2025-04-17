from base_model import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class TransferLearningModel(BaseModel):
    def __init__(self, train=False, base_model_name='efficientnet'):
        super().__init__()
        self.base_model_name = base_model_name
        if train:
            self.train_model()
        else:
            try:
                self.load_model()
                print(f"Transfer Learning model ({self.base_model_name}) loaded!")
            except:
                self.train_model()
                print(f"Transfer Learning model ({self.base_model_name}) trained!")

    def load_model(self):
        self.model = tf.keras.models.load_model(f'transfer_learning_model_{self.base_model_name}.keras')
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

        # Convert grayscale to RGB by repeating the channel 3 times
        # This is necessary because pre-trained models expect RGB input
        images_rgb = np.repeat(images.reshape(images.shape[0], images.shape[1], images.shape[2], 1), 3, axis=3)

        # Resize images to match the input size expected by the pre-trained model
        input_size = (224, 224)  # Standard input size for many pre-trained models
        images_resized = np.array([tf.image.resize(img, input_size).numpy() for img in images_rgb])

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(images_resized, labels, test_size=0.2, random_state=42)

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.2,
            horizontal_flip=False,  # Don't flip characters horizontally
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        # Create base model
        if self.base_model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif self.base_model_name == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:  # default to EfficientNet
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model layers
        base_model.trainable = False

        # Create the model
        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.label_to_index), activation='softmax')
        ])

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Add callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )

        # Train the model with frozen base layers
        print("Training with frozen base layers...")
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=20,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr]
        )

        # Unfreeze some layers of the base model for fine-tuning
        if self.base_model_name == 'resnet50':
            # Unfreeze the last 30 layers
            for layer in base_model.layers[-30:]:
                layer.trainable = True
        elif self.base_model_name == 'mobilenet':
            # Unfreeze the last 20 layers
            for layer in base_model.layers[-20:]:
                layer.trainable = True
        else:  # EfficientNet
            # Unfreeze the last 15 layers
            for layer in base_model.layers[-15:]:
                layer.trainable = True

        # Recompile the model with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Fine-tune the model
        print("Fine-tuning the model...")
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=16),  # Smaller batch size for fine-tuning
            epochs=30,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr]
        )

        # Save the model
        self.model.save(f'transfer_learning_model_{self.base_model_name}.keras')
        print(f"Transfer Learning model ({self.base_model_name}) saved!")

    def scan_img(self, img, return_confidence=False, return_top_k=False, k=3):
        # Convert grayscale to RGB by repeating the channel 3 times
        img_rgb = np.repeat(img, 3, axis=3)

        # Resize to the input size expected by the pre-trained model
        input_size = (224, 224)
        img_resized = tf.image.resize(img_rgb, input_size)

        # Make prediction
        prediction = self.model.predict(img_resized)
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
    # Train and evaluate models with different base architectures
    for base_model_name in ['efficientnet', 'mobilenet', 'resnet50']:
        print(f"\n{'='*50}")
        print(f"TRAINING AND EVALUATING {base_model_name.upper()}")
        print(f"{'='*50}")
        model = TransferLearningModel(train=True, base_model_name=base_model_name)
        accuracy = model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)
        print(f"{base_model_name.upper()} Accuracy: {accuracy:.2f}%")