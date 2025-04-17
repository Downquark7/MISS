from base_model import BaseModel
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        print("Training RandomForest model...")
        self.train_model()
        print("RandomForest model trained!")

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
            # Flatten the image for Random Forest
            images.append(img.flatten() / 255.0)
            labels.append(self.label_to_index[label])

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=None,    # Maximum depth of trees
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            n_jobs=-1,         # Use all available cores
            random_state=42
        )
        self.model.fit(X_scaled, y)

    def scan_img(self, img):
        # Flatten the image
        img_flat = img.flatten()
        
        # Scale features
        img_scaled = self.scaler.transform(img_flat.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(img_scaled)
        character = self.index_to_label[prediction[0]]
        
        return str(character)


if __name__ == "__main__":
    model = RandomForestModel()
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)