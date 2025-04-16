from base_model import BaseModel
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier


class KNN(BaseModel):
    def __init__(self):
        super().__init__()
        print("Training NaiveKNN model...")
        self.train_model()
        print("NaiveKNN model trained!")

    @staticmethod
    def extract_features(img):
        features = [np.average(img), np.sum(np.abs(img - np.rot90(img))), np.sum(np.abs(img - np.rot90(img, 2))),
                    np.sum(np.abs(img - np.rot90(img, 3))), np.sum(np.abs(img - np.fliplr(img))),
                    np.sum(np.abs(img - np.flipud(img))), np.sum(np.abs(img - np.fliplr(img) - np.flipud(img))),
                    np.sum(np.abs(img - np.rot90(img, 1) - np.rot90(img, 3))),
                    np.sum(np.abs(img - np.rot90(img, 1) - np.fliplr(img))),
                    np.sum(np.abs(img - np.rot90(img, 1) - np.flipud(img))),
                    np.sum(np.abs(img - np.rot90(img, 2) - np.fliplr(img)))]

        return features

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
        feature_list = []
        labels = []
        for filename, label in zip(self.labels_df['filename'], self.labels_df['label']):
            img_path = os.path.join(image_folder, filename)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = self.pad_image(img_array)
            images.append(img / 255.0)
            labels.append(self.label_to_index[label])
            feature_list.append(self.extract_features(img))

        # Train KNN model
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(feature_list, labels)

    def scan_img(self, img):
        features = np.array(self.extract_features(img)).reshape(1, -1)
        print(features)
        prediction = self.model.predict(features)
        character = self.index_to_label[prediction[0]]
        return str(character)


if __name__ == "__main__":
    model = KNN()
    # model.scan_img_path('0_)_test_images/IMG_8500.jpg')
    # model.train_model()
    # model.load_model()
    model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)
