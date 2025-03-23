import numpy as np
import os
import cv2
from matplotlib import pyplot as plt


class BaseModel:
    def __init__(self):
        self.image_size = (28, 28)

    def process_folder(self, folder_path):

        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for img_path in file_paths:
            print("Opening: ", img_path)
            self.process_file(img_path)

    def process_file(self, img_path):
        self.scan_img_path(img_path)

    def scan_img(self, img):
        pass

    def scan_img_path(self, img_path, plot=True):
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 239, 17)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_list = ""
        contours = sorted(contours,
                          key=lambda c: cv2.boundingRect(c)[0])  # Sort contours by x-coordinate of bounding rectangle
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x1, y1, x2, y2 = x, y, x + w, y + h
            mask = np.zeros_like(gray)  # Create a blank mask
            cv2.drawContours(mask, [c], -1, 255, -1)
            binary_region = cv2.bitwise_and(binary, binary, mask=mask)[y1:y2, x1:x2]
            # boxes.append((x, y, x + w, y + h, binary_region))
            padded_img = self.pad_image(binary_region)
            img = np.array(padded_img) / 255.0
            img = img.reshape(1, 28, 28, 1)
            character = self.scan_img(img)
            char_list += character
            # plt.title(f"Predicted: {character}")
            # plt.imshow(tf.keras.preprocessing.image.array_to_img(padded_img), cmap='gray')
            # plt.show()
        print(char_list)
        if plot:
            plt.imshow(binary, cmap="gray")
            plt.title(f"{char_list}")
            plt.show()

    def pad_image(self, img_array):
        # Get the original dimensions
        original_height, original_width = img_array.shape[:2]
        target_width, target_height = self.image_size

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height
        target_aspect_ratio = target_width / target_height

        # Resize the image while maintaining the aspect ratio
        if aspect_ratio > target_aspect_ratio:
            # Fit to width
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Fit to height
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        resized_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

        padded_img = np.zeros((target_height, target_width), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

        return padded_img
