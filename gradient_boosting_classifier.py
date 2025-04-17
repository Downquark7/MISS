from base_model import BaseModel
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class GradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__()
        print("Training Gradient Boosting model...")
        self.train_model()
        print("Gradient Boosting model trained!")

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
            # Flatten the image for Gradient Boosting
            images.append(img.flatten() / 255.0)
            labels.append(self.label_to_index[label])

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Gradient Boosting model
        # Using a smaller number of estimators and max_depth for faster training
        self.model = GradientBoostingClassifier(
            n_estimators=50,        # Number of boosting stages
            learning_rate=0.1,      # Shrinks the contribution of each tree
            max_depth=3,            # Maximum depth of the individual trees
            min_samples_split=5,    # Minimum samples required to split a node
            min_samples_leaf=2,     # Minimum samples required at a leaf node
            subsample=0.8,          # Fraction of samples used for fitting the trees
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
            # For Gradient Boosting, we can use the probability of the predicted class
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
    model = GradientBoostingModel()
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)