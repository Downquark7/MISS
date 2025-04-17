from base_model import BaseModel
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNN(BaseModel):
    """KNN-based classifier for character recognition.

    This classifier uses a K-Nearest Neighbors algorithm with custom feature
    extraction for character recognition.
    """

    def __init__(self):
        """Initialize the KNN classifier and train the model."""
        super().__init__()
        print("Training KNN model...")
        self.train_model()
        print("KNN model trained!")

    @staticmethod
    def extract_features(img):
        """Extract features from an image for KNN classification.

        This method extracts various features from the image, including:
        - Symmetry features
        - Zoning features
        - Projection histograms
        - Gradient features
        - Moment-based features
        - Contour-based features

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: A feature vector representing the image.
        """
        # Ensure image is normalized
        if img.max() > 1.0:
            img = img / 255.0

        # Original symmetry features
        symmetry_features = [
            np.average(img), 
            np.sum(np.abs(img - np.rot90(img))), 
            np.sum(np.abs(img - np.rot90(img, 2))),
            np.sum(np.abs(img - np.rot90(img, 3))), 
            np.sum(np.abs(img - np.fliplr(img))),
            np.sum(np.abs(img - np.flipud(img))), 
            np.sum(np.abs(img - np.fliplr(img) - np.flipud(img))),
            np.sum(np.abs(img - np.rot90(img, 1) - np.rot90(img, 3))),
            np.sum(np.abs(img - np.rot90(img, 1) - np.fliplr(img))),
            np.sum(np.abs(img - np.rot90(img, 1) - np.flipud(img))),
            np.sum(np.abs(img - np.rot90(img, 2) - np.fliplr(img)))
        ]

        # Zoning features - divide image into 5x5 zones for more detail
        h, w = img.shape
        zone_h, zone_w = h // 5, w // 5
        zoning_features = []
        for i in range(5):
            for j in range(5):
                zone = img[i*zone_h:min((i+1)*zone_h, h), j*zone_w:min((j+1)*zone_w, w)]
                zoning_features.append(np.mean(zone))
                # Add standard deviation for each zone to capture texture
                zoning_features.append(np.std(zone))

        # Projection histograms (horizontal and vertical)
        h_proj = np.sum(img, axis=1) / w  # Horizontal projection
        v_proj = np.sum(img, axis=0) / h  # Vertical projection

        # Add diagonal projections for better capturing of slashes and diagonals
        # Top-left to bottom-right diagonal
        diag1 = []
        for i in range(-h+1, w):
            diag = np.diagonal(img, offset=i)
            diag1.append(np.mean(diag))

        # Top-right to bottom-left diagonal
        diag2 = []
        for i in range(-h+1, w):
            diag = np.diagonal(np.fliplr(img), offset=i)
            diag2.append(np.mean(diag))

        # Gradient features
        # Compute horizontal and vertical gradients
        h_gradient = np.abs(img[:, 1:] - img[:, :-1])
        v_gradient = np.abs(img[1:, :] - img[:-1, :])

        # Compute gradient statistics
        gradient_features = [
            np.mean(h_gradient),
            np.mean(v_gradient),
            np.std(h_gradient),
            np.std(v_gradient),
            # Add more gradient features
            np.max(h_gradient),
            np.max(v_gradient),
            np.sum(h_gradient),
            np.sum(v_gradient)
        ]

        # Moment-based features
        # Central moments
        y_indices, x_indices = np.mgrid[:h, :w]
        x_mean = np.sum(x_indices * img) / np.sum(img) if np.sum(img) > 0 else w/2
        y_mean = np.sum(y_indices * img) / np.sum(img) if np.sum(img) > 0 else h/2

        # Compute central moments
        moment_features = []
        for p in range(4):  # Increase to 4th order
            for q in range(4):
                if p + q <= 4 and p + q > 0:  # Skip 0,0 and limit to 4th order
                    moment = np.sum(((x_indices - x_mean) ** p) * ((y_indices - y_mean) ** q) * img)
                    moment_features.append(moment)

        # Add Hu moments which are invariant to translation, scale, and rotation
        # These are particularly useful for character recognition
        m00 = np.sum(img)
        if m00 > 0:
            m10 = np.sum(x_indices * img)
            m01 = np.sum(y_indices * img)
            mu20 = np.sum(((x_indices - x_mean) ** 2) * img) / m00
            mu02 = np.sum(((y_indices - y_mean) ** 2) * img) / m00
            mu11 = np.sum((x_indices - x_mean) * (y_indices - y_mean) * img) / m00
            mu30 = np.sum(((x_indices - x_mean) ** 3) * img) / m00
            mu03 = np.sum(((y_indices - y_mean) ** 3) * img) / m00
            mu12 = np.sum(((x_indices - x_mean) ** 1) * ((y_indices - y_mean) ** 2) * img) / m00
            mu21 = np.sum(((x_indices - x_mean) ** 2) * ((y_indices - y_mean) ** 1) * img) / m00

            # First Hu moment (invariant to translation, scale, rotation)
            hu1 = mu20 + mu02
            # Second Hu moment
            hu2 = (mu20 - mu02)**2 + 4*(mu11**2)
            # Third Hu moment
            hu3 = (mu30 - 3*mu12)**2 + (3*mu21 - mu03)**2

            hu_moments = [hu1, hu2, hu3]
        else:
            hu_moments = [0, 0, 0]

        # Add contour-based features
        # Create a binary image for contour detection
        binary = (img > 0.5).astype(np.uint8)
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Contour area
                area = cv2.contourArea(largest_contour)
                # Contour perimeter
                perimeter = cv2.arcLength(largest_contour, True)
                # Contour circularity
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                # Contour aspect ratio
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                # Contour extent
                rect_area = w * h
                extent = float(area) / rect_area if rect_area > 0 else 0

                contour_features = [area, perimeter, circularity, aspect_ratio, extent]
            else:
                contour_features = [0, 0, 0, 0, 0]
        except:
            contour_features = [0, 0, 0, 0, 0]

        # Combine all features
        features = (symmetry_features + zoning_features + list(h_proj) + list(v_proj) + 
                   diag1 + diag2 + gradient_features + moment_features + hu_moments + contour_features)

        # Normalize features to have similar scales
        features = np.array(features)
        # Replace NaN values with 0
        features = np.nan_to_num(features)

        return features

    def train_model(self, labels_file='labels.csv', image_folder='labeled_characters_binary',
                  label_mappings_path='label_mappings.npy', n_neighbors=3):
        """Train the KNN model.

        Args:
            labels_file (str, optional): Path to the CSV file containing labels. Defaults to 'labels.csv'.
            image_folder (str, optional): Path to the folder containing images. Defaults to 'labeled_characters_binary'.
            label_mappings_path (str, optional): Path to save label mappings. Defaults to 'label_mappings.npy'.
            n_neighbors (int, optional): Number of neighbors for KNN. Defaults to 3.
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
        feature_list = []
        labels = []
        for filename, label in zip(self.labels_df['filename'], self.labels_df['label']):
            img_path = os.path.join(image_folder, filename)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = self.pad_image(img_array)
            images.append(img / 255.0)
            labels.append(self.label_to_index[label])
            feature_list.append(self.extract_features(img))

        # Convert feature list to numpy array for preprocessing
        feature_array = np.array(feature_list)

        # Replace any NaN or infinite values
        feature_array = np.nan_to_num(feature_array)

        # Normalize features to have zero mean and unit variance
        # This helps ensure all features contribute equally to the distance calculations
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_array)

        # Train KNN model with optimized parameters
        # Use Euclidean distance which often works well for normalized features
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            metric='euclidean',
            algorithm='auto'
        )
        self.model.fit(scaled_features, labels)

        # Save the scaler for later use during prediction
        self.scaler = scaler

    def scan_img(self, img):
        """Scan an image and predict the character.

        Args:
            img (numpy.ndarray): The image to scan.

        Returns:
            str: The predicted character.
        """
        # Reshape the 4D tensor (1, 28, 28, 1) to 2D image (28, 28)
        img_2d = img.reshape(28, 28)

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
    """Main entry point for testing the KNN model."""
    model = KNN()
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)
