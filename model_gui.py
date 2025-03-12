import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import cv2

# Model and label loading
model = tf.keras.models.load_model('character_model.keras')
index_to_label = np.load('label_mappings.npy', allow_pickle=True).item()

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Character Recognizer")

        # Drawing variables
        self.last_x = None
        self.last_y = None
        self.draw_color = "black"
        self.background_color = "white"
        self.line_width = 10

        # Create canvas
        self.canvas = tk.Canvas(root, bg=self.background_color, width=280, height=280)
        self.canvas.pack(pady=20)

        # Create drawing image
        self.image = Image.new("L", (280, 280), self.background_color)
        self.draw = ImageDraw.Draw(self.image)

        # Setup mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.result_window = None  # Track the result window
        self.predict_btn = tk.Button(
            btn_frame, text="Predict", command=self.predict_drawing
        )
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        self.clear_btn = tk.Button(
            btn_frame, text="Clear", command=self.clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=self.line_width, fill=self.draw_color,
                capstyle=tk.ROUND, smooth=tk.TRUE
            )
            # Draw on image
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill="black",  # Use black for drawing (will invert later)
                width=self.line_width
            )
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), self.background_color)
        self.draw = ImageDraw.Draw(self.image)

    def preprocess_image(self):
        # Invert colors (model expects white-on-black)
        inverted = ImageOps.invert(self.image)

        contours, _ = cv2.findContours(np.array(inverted), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            inverted = inverted.crop((x, y, x + w, y + h))

        # Resize to match model input size
        img = inverted.resize((28, 28))

        # cv2.imshow("Image", np.array(img))
        # cv2.waitKey(0)

        # Convert to array and normalize
        img_array = np.array(img) / 255.0

        # Add batch and channel dimensions
        return img_array.reshape(1, 28, 28, 1)

    def predict_drawing(self):
        # Preprocess drawn image
        processed_img = self.preprocess_image()

        # Make prediction
        prediction = model.predict(processed_img)
        predicted_index = np.argmax(prediction)
        character = index_to_label[predicted_index]

        # Show result
        if self.result_window is None or not self.result_window.winfo_exists():
            self.result_window = tk.Toplevel(self.root)
            self.result_window.title("Prediction Result")
            self.result_label = tk.Label(self.result_window, font=("Arial", 16))
            self.result_label.pack(padx=20, pady=20)

        self.result_label.config(text=f"Predicted Character: {character}")
        self.result_window.lift()


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
