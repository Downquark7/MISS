import os
import re
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import pandas as pd


class CharacterLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Math Expression Labeler")

        # Initialize variables
        self.image_paths = []
        self.current_idx = 0
        self.boxes = []
        self.labels = []
        self.selected_boxes = set()
        self.output_dir = "labeled_characters"
        self.label_csv = "labels.csv"

        # Setup UI
        self.create_widgets()
        self.bind_events()

        # Initialize CSV
        if not os.path.exists(self.label_csv):
            pd.DataFrame(columns=["filename", "label"]).to_csv(self.label_csv, index=False)

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=1200, height=800)
        self.canvas.pack()

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        self.prev_btn = tk.Button(control_frame, text="<< Previous", command=self.prev_image)
        self.next_btn = tk.Button(control_frame, text="Next >>", command=self.next_image)
        self.save_btn = tk.Button(control_frame, text="Save Labels", command=self.save_labels)

        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.status = tk.Label(self.root, text="")
        self.status.pack()

    def bind_events(self):
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<space>", lambda e: self.save_labels())
        self.canvas.bind("<Button-1>", self.on_click)

    def load_folder(self, folder_path):
        self.image_paths = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda x: int(re.search(r'(\d+)', x).group(1)))
        self.current_idx = 0
        self.process_and_show_image()

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 55, 25)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 3 < w < 200 and 3 < h < 150:
                boxes.append((x, y, x + w, y + h))

        return image, sorted(boxes, key=lambda b: b[0])

    def process_and_show_image(self):
        if self.current_idx < len(self.image_paths):
            image_path = self.image_paths[self.current_idx]
            self.original_image, self.boxes = self.process_image(image_path)
            self.selected_boxes = set()
            self.show_processed_image()
            self.status.config(
                text=f"Image {self.current_idx + 1}/{len(self.image_paths)} - Select boxes to combine or press Save")

    def show_processed_image(self):
        display_image = self.original_image.copy()
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            color = (0, 255, 0) if i in self.selected_boxes else (0, 0, 255)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

        img = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_click(self, event):
        x, y = event.x, event.y
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                if i in self.selected_boxes:
                    self.selected_boxes.remove(i)
                else:
                    self.selected_boxes.add(i)
                self.show_processed_image()
                break

    def combine_selected_boxes(self):
        if len(self.selected_boxes) >= 2:
            selected = [self.boxes[i] for i in self.selected_boxes]
            x1 = min(b[0] for b in selected)
            y1 = min(b[1] for b in selected)
            x2 = max(b[2] for b in selected)
            y2 = max(b[3] for b in selected)

            new_boxes = [b for i, b in enumerate(self.boxes) if i not in self.selected_boxes]
            new_boxes.append((x1, y1, x2, y2))
            self.boxes = sorted(new_boxes, key=lambda b: b[0])
            self.selected_boxes = set()
            self.show_processed_image()

    def save_labels(self):
        if not self.boxes:
            return

        base_name = os.path.basename(self.image_paths[self.current_idx])
        file_id = os.path.splitext(base_name)[0]

        labels = []
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            char_img = self.original_image[y1:y2, x1:x2]
            if char_img.size == 0:
                continue

            label = simpledialog.askstring("Label Input",
                                           f"Label for character {i + 1} (Press Esc to skip):",
                                           parent=self.root)

            if label:
                filename = f"{file_id}_char_{i}.png"
                cv2.imwrite(os.path.join(self.output_dir, filename),
                            cv2.cvtColor(char_img, cv2.COLOR_RGB2BGR))
                labels.append({"filename": filename, "label": label})

        if labels:
            pd.DataFrame(labels).to_csv(self.label_csv, mode='a',
                                        header=not os.path.exists(self.label_csv),
                                        index=False)

        self.next_image()

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.process_and_show_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.process_and_show_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = CharacterLabeler(root)

    # Ask for input folder
    folder_path = filedialog.askdirectory(title="Select Folder with Math Expression Images")
    if folder_path:
        os.makedirs(app.output_dir, exist_ok=True)
        app.load_folder(folder_path)

    root.mainloop()