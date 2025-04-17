from resized_base_model import ResizedBaseModel
from KNN import KNN
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier

class ResizedKNN(ResizedBaseModel):
    def __init__(self):
        super().__init__()
        print("Training Resized KNN model (14x14)...")
        self.train_model()
        print("Resized KNN model trained!")

    # Use the extract_features method from KNN
    @staticmethod
    def extract_features(img):
        return KNN.extract_features(img)

    def train_model(self):
        # Load and preprocess data
        self.labels_df = pd.read_csv('labels.csv')
        image_folder = 'labeled_characters_binary'

        # Create label mappings
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(self.labels_df['label']))}
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        # Save label mappings for later use
        np.save('label_mappings_resized.npy', self.index_to_label)

        # Load images and labels
        images = []
        feature_list = []
        labels = []
        for filename, label in zip(self.labels_df['filename'], self.labels_df['label']):
            img_path = os.path.join(image_folder, filename)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = self.pad_image(img_array)  # This will pad to 14x14 due to ResizedBaseModel
            images.append(img / 255.0)
            labels.append(self.label_to_index[label])
            feature_list.append(self.extract_features(img))

        # Convert feature list to numpy array for preprocessing
        feature_array = np.array(feature_list)

        # Replace any NaN or infinite values
        feature_array = np.nan_to_num(feature_array)

        # Normalize features to have zero mean and unit variance
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_array)

        # Train KNN model with optimized parameters
        self.model = KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            metric='euclidean',
            algorithm='auto'
        )
        self.model.fit(scaled_features, labels)

    def scan_img(self, img):
        # Reshape the 4D tensor (1, 14, 14, 1) to 2D image (14, 14)
        img_2d = img.reshape(14, 14)

        # Extract features
        features = np.array(self.extract_features(img_2d)).reshape(1, -1)

        # Replace any NaN or infinite values
        features = np.nan_to_num(features)

        # Scale features using the same scaler used during training
        scaled_features = self.scaler.transform(features)

        # Make prediction
        prediction = self.model.predict(scaled_features)
        character = self.index_to_label[prediction[0]]

        return str(character)

if __name__ == "__main__":
    model = ResizedKNN()
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)
