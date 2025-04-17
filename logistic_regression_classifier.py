from base_model import BaseModel
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        print("Training Logistic Regression model...")
        self.train_model()
        print("Logistic Regression model trained!")

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
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = self.pad_image(img_array)
            # Flatten the image for Logistic Regression
            images.append(img.flatten() / 255.0)
            labels.append(self.label_to_index[label])

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Logistic Regression model
        # Using L2 regularization and increased max_iter for convergence
        self.model = LogisticRegression(
            C=1.0,                # Inverse of regularization strength
            penalty='l2',         # L2 regularization
            solver='lbfgs',       # Efficient solver for multiclass problems
            multi_class='multinomial',  # Multinomial loss for multiclass
            max_iter=1000,        # Increased max iterations for convergence
            random_state=42
        )
        self.model.fit(X_scaled, y)

    def scan_img(self, img, return_confidence=False):
        # Flatten the image
        img_flat = img.flatten()
        
        # Scale features
        img_scaled = self.scaler.transform(img_flat.reshape(1, -1))
        
        # Make prediction
        if return_confidence:
            # For Logistic Regression, we can use the probability of the predicted class
            # as a confidence measure
            proba = self.model.predict_proba(img_scaled)[0]
            prediction = self.model.predict(img_scaled)[0]
            character = self.index_to_label[prediction]
            confidence = proba[prediction]
            return character, confidence
        else:
            prediction = self.model.predict(img_scaled)
            character = self.index_to_label[prediction[0]]
            return str(character)


if __name__ == "__main__":
    model = LogisticRegressionModel()
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)