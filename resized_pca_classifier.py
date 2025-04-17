from resized_base_model import ResizedBaseModel
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ResizedPCAClassifier(ResizedBaseModel):
    def __init__(self):
        super().__init__()
        print("Training Resized PCA-KNN model (14x14)...")
        self.train_model()
        print("Resized PCA-KNN model trained!")

    def train_model(self):
        # Load and preprocess data
        self.labels_df = pd.read_csv('labels.csv')
        image_folder = 'labeled_characters_binary'

        # Create label mappings
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(self.labels_df['label']))}
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        # Save label mappings for later use
        np.save('label_mappings_resized_pca.npy', self.index_to_label)

        # Load images and labels
        images = []
        labels = []
        for filename, label in zip(self.labels_df['filename'], self.labels_df['label']):
            img_path = os.path.join(image_folder, filename)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = self.pad_image(img_array)  # This will pad to 14x14 due to ResizedBaseModel
            # Flatten the image for PCA
            images.append(img.flatten() / 255.0)
            labels.append(self.label_to_index[label])

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Create a pipeline with StandardScaler, PCA, and KNN
        # Use fewer principal components (25 instead of 50) since the image is smaller
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=25, random_state=42)),  # Reduce to 25 principal components
            ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance'))
        ])

        # Train the pipeline
        self.pipeline.fit(X, y)

        # Store the components for visualization if needed
        self.pca_components = self.pipeline.named_steps['pca'].components_
        self.explained_variance = self.pipeline.named_steps['pca'].explained_variance_ratio_
        print(f"PCA explained variance: {sum(self.explained_variance):.2f}")

    def scan_img(self, img, return_confidence=False):
        # Flatten the image
        img_flat = img.flatten()
        
        # Use the pipeline to make a prediction
        prediction = self.pipeline.predict(img_flat.reshape(1, -1))
        character = self.index_to_label[prediction[0]]
        
        if return_confidence:
            # For KNN, we can use the distance to the nearest neighbors as a confidence measure
            # Get distances to nearest neighbors
            distances, indices = self.pipeline.named_steps['knn'].kneighbors(
                self.pipeline.named_steps['pca'].transform(
                    self.pipeline.named_steps['scaler'].transform(img_flat.reshape(1, -1))
                )
            )
            
            # Convert distance to confidence (closer = more confident)
            # Use the inverse of the average distance to the k nearest neighbors
            avg_distance = np.mean(distances[0])
            confidence = 1.0 / (1.0 + avg_distance)  # Scale to [0, 1]
            
            return character, confidence
        else:
            return str(character)


if __name__ == "__main__":
    model = ResizedPCAClassifier()
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)