from base_model import BaseModel
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

class ResizedBaseModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.image_size = (14, 14)  # Override the image size to be 14x14

    def scan_img_path(self, img_path, plot=True):
        """
        Override the scan_img_path method to use 14x14 images instead of 28x28.
        Most of this code is copied from BaseModel.scan_img_path, with the reshape modified.
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
                          key=lambda c: cv2.boundingRect(c)[0])  # Sort contours by x-coordinate
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
            # This is the key change: reshape to 14x14 instead of 28x28
            img = img.reshape(1, 14, 14, 1)
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
