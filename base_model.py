import numpy as np
import os
import cv2
from matplotlib import pyplot as plt


class BaseModel:
    """Base class for all character recognition models.

    This class provides common functionality for image processing, evaluation,
    and character recognition that is shared across all model implementations.
    """

    def __init__(self):
        """Initialize the base model with default image size."""
        self.image_size = (28, 28)

    def process_folder(self, folder_path):
        """Process all images in a folder.

        Args:
            folder_path (str): Path to the folder containing images to process.
        """
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for img_path in file_paths:
            print("Opening: ", img_path)
            self.scan_img_path(img_path)

    def eval_folder(self, folder_path, char_list, plot=True):
        """Evaluate model performance on a folder of images.

        Args:
            folder_path (str): Path to the folder containing images to evaluate.
            char_list (str): String of expected characters in the images, in order.
            plot (bool, optional): Whether to plot the images with bounding boxes. Defaults to True.

        Returns:
            tuple: (overall_accuracy, per_char_accuracy) where:
                - overall_accuracy (float): Overall accuracy percentage.
                - per_char_accuracy (dict): Dictionary mapping characters to their accuracy percentages.
        """
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        total_chars = 0
        scanned_chars = 0
        correct_matches = 0
        char_accuracy = {char: {"total": 0, "correct": 0} for char in set(char_list)}

        for img_path in file_paths:
            print("Opening: ", img_path)
            read_chars = self.scan_img_path(img_path, plot=plot)
            scanned_chars += len(read_chars)  # Increment scanned_chars by the number of characters read
            for rc, cl in zip(read_chars, char_list):
                total_chars += 1
                if rc == cl:
                    correct_matches += 1
                    char_accuracy[cl]["correct"] += 1
                char_accuracy[cl]["total"] += 1

        overall_accuracy = (correct_matches / total_chars) * 100 if total_chars > 0 else 0
        per_char_accuracy = {char: (data["correct"] / data["total"] * 100 if data["total"] > 0 else 0)
                             for char, data in char_accuracy.items()}

        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"Per Character Accuracy: { {char: f"{accuracy:.2f}%" for char, accuracy in per_char_accuracy.items()} }")
        print(f"Scanned Character Count: {scanned_chars}")
        print(f"Expected Character Count: {total_chars}")
        return overall_accuracy, per_char_accuracy

    def scan_img(self, img):
        """Scan an image and predict the character.

        This is an abstract method that should be implemented by subclasses.

        Args:
            img (numpy.ndarray): The image to scan.

        Returns:
            str: The predicted character.
        """
        pass

    def scan_img_path(self, img_path, plot=True):
        """Scan an image file, detect characters, and predict them.

        This method reads an image file, preprocesses it, detects character regions,
        and uses the scan_img method to predict each character.

        Args:
            img_path (str): Path to the image file.
            plot (bool, optional): Whether to plot the image with bounding boxes. Defaults to True.

        Returns:
            str: String of predicted characters.
        """
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 239, 17)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_list = ""
        binary_regions = []
        contours = sorted(contours,
                          key=lambda c: cv2.boundingRect(c)[0])  # Sort contours by x-coordinate of bounding rectangle
        bounding_boxes = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 5 < h < 1000 and 5 < w < 1000:
                bounding_boxes.append((x, y, x + w, y + h))

        # Combine overlapping regions
        merged_boxes = []
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            merged = False
            for i, (mx1, my1, mx2, my2) in enumerate(merged_boxes):
                if max(x1, mx1) <= min(x2 * 1.01, mx2 * 1.01):
                    merged_boxes[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                    merged = True
                    break
            if not merged:
                merged_boxes.append((x1, y1, x2, y2))

        for mx1, my1, mx2, my2 in merged_boxes:
            mask = np.zeros_like(gray)  # Create a blank mask
            cv2.rectangle(mask, (mx1, my1), (mx2, my2), 255, -1)
            binary_region = cv2.bitwise_and(binary, binary, mask=mask)[my1:my2, mx1:mx2]
            binary_regions.append(binary_region)

        for binary_region in binary_regions:
            padded_img = self.pad_image(binary_region)
            img = np.array(padded_img) / 255.0
            img = img.reshape(1, 28, 28, 1)
            character = self.scan_img(img)
            char_list += character

        print(char_list)
        if plot:
            thickness = max(1, int(gray.shape[0] * 0.02))
            for mx1, my1, mx2, my2 in merged_boxes:
                cv2.rectangle(binary, (mx1, my1), (mx2, my2), (255, 255, 255), thickness)
            plt.imshow(binary, cmap="gray")
            plt.title(f"{char_list}")
            plt.show()
        return char_list

    def pad_image(self, img_array):
        """Resize and pad an image to the target size while maintaining aspect ratio.

        Args:
            img_array (numpy.ndarray): The input image array.

        Returns:
            numpy.ndarray: The resized and padded image.
        """
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
            new_height = max(1, int(target_width / aspect_ratio))
        else:
            # Fit to height
            new_height = target_height
            new_width = max(1, int(target_height * aspect_ratio))

        resized_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

        padded_img = np.zeros((target_height, target_width), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

        return padded_img
