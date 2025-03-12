import os
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def scan_image(img_path):
    output_dir = "labeled_characters"
    label_csv = "labels.csv"

    if os.path.exists(label_csv):
        last_row = pd.read_csv(label_csv).iloc[-1]
        file_id = int(re.search(r'(\d+)', last_row['filename']).group(1)) + 1
    else:
        file_id = 1

    labels = []

    image = cv2.imread(img_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = gray
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    # b_range = range(1, 25, 2)
    # i_range = range(3, 256, 4)
    # j_range = range(1, 100, 4)
    b_range = [25]
    i_range = [239]
    j_range = [17]
    target_count = 10
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

    if good_count != target_count or len(contours) != target_count:
        print(f"Skipping: {img_path}")
        return

    contours = sorted(contours,
                      key=lambda c: cv2.boundingRect(c)[0])  # Sort contours by x-coordinate of bounding rectangle

    i = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x1, y1, x2, y2 = x, y, x + w, y + h
        mask = np.zeros_like(gray)  # Create a blank mask
        cv2.drawContours(mask, [c], -1, 255, -1)
        binary_region = cv2.bitwise_and(binary, binary, mask=mask)
        char_img = image[y1:y2, x1:x2]
        char_img_binary = binary_region[y1:y2, x1:x2]
        label = str(i)
        filename = f"{file_id}_char_{i}.png"
        cv2.imwrite(os.path.join(output_dir, filename), char_img)
        cv2.imwrite(os.path.join(output_dir + "_binary", filename), char_img_binary)
        labels.append({"filename": filename, "label": label})
        i += 1

    if labels:
        pd.DataFrame(labels).to_csv(label_csv, mode='a',
                                    header=not os.path.exists(label_csv),
                                    index=False)
        os.remove(img_path)


def numerical_sort(file):
    match = re.search(r'(\d+)', file)
    return int(match.group(1)) if match else float('inf')


if __name__ == "__main__":
    folder_path = '0_9_new_data'

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for img_path in file_paths:
    print("Opening: ", img_path)
    scan_image(img_path)
