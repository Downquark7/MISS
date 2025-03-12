import os
import random
import re
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def pad_image(img_array, target_size):
    # Get the original dimensions
    original_height, original_width = img_array.shape
    target_width, target_height = target_size

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

    resized_img = tf.image.resize(tf.expand_dims(img_array, axis=-1), [new_height, new_width])

    # Create a black background image of the target size
    padded_img = tf.image.pad_to_bounding_box(
        resized_img,
        offset_height=(target_height - new_height) // 2,
        offset_width=(target_width - new_width) // 2,
        target_height=target_height,
        target_width=target_width
    )

    return padded_img


if __name__ == "__main__":
    # Model and label loading
    model = tf.keras.models.load_model('character_model.keras')
    index_to_label = np.load('label_mappings.npy', allow_pickle=True).item()
    labels_df = pd.read_csv('labels.csv')
    label_to_index = {label: idx for idx, label in enumerate(np.unique(labels_df['label']))}


def scan_image(img_path, b0, i0, j0, target_count):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = gray
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    good_count = 0
    b_range = range(1, 25, 2) if b0 == 0 else [b0]
    i_range = range(3, 256, 4) if i0 == 0 else [i0]
    j_range = range(1, 100, 4) if j0 == 0 else [j0]

    for b in b_range:
        if b > 1:
            blur = cv2.GaussianBlur(gray, (b, b), 0)
        for i in i_range:
            for j in j_range:
                good_count = 0
                binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, i, j)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if 10 < h < 1000 and 5 < w < 1000:
                        good_count += 1
                if good_count == target_count == len(contours):
                    print(b, i, j)
                    # plt.imshow(binary, cmap='gray')
                    # plt.show()
                    break
            if target_count == 0 or good_count == target_count == len(contours):
                break
        if target_count == 0 or good_count == target_count == len(contours):
            break

    # blur = cv2.GaussianBlur(gray, (15, 15), 0)
    # binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 200, 50)
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        padded_img = pad_image(binary_region, target_size=(28, 28))
        img = np.array(padded_img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)
        character = index_to_label[predicted_index]
        char_list += character
        # plt.title(f"Predicted: {character}")
        # plt.imshow(tf.keras.preprocessing.image.array_to_img(padded_img), cmap='gray')
        # plt.show()

    plt.imshow(binary, cmap="gray")
    plt.title(f"{char_list}")
    plt.show()


def numerical_sort(file):
    match = re.search(r'(\d+)', file)
    return int(match.group(1)) if match else float('inf')


if __name__ == "__main__":
    import argparse

    folder_path = 'test_images'
    parser = argparse.ArgumentParser(description="Scan images with optional parameters.")
    parser.add_argument("filename", nargs="?", help="Filename of the image to scan.")
    parser.add_argument("-b", type=int, default=25, help="Blur kernel size.")
    parser.add_argument("-i", type=int, default=239, help="Parameter i for adaptive thresholding.")
    parser.add_argument("-j", type=int, default=17, help="Parameter j for adaptive thresholding.")
    parser.add_argument("-c", type=int, default=0, help="Target contour count.")

    args = parser.parse_args()
    b = args.b
    i = args.i
    j = args.j
    c = args.c

    if args.filename:  # Check if a filename argument is provided
        img_path = os.path.join(folder_path, args.filename)
        if os.path.isfile(img_path):
            print("Opening: ", img_path)
            scan_image(img_path, b, i, j, c)
        else:
            print(f"File {args.filename} not found in {folder_path}")
    else:
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for img_path in file_paths:
            print("Opening: ", img_path)
            scan_image(img_path, b, i, j, c)
