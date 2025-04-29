# MISS - Math Identification and Solving Systems

MISS is a machine learning project for recognizing handwritten mathematical expressions and symbols. The system can identify individual characters from images containing mathematical expressions and provides a user-friendly interface for drawing and recognizing characters in real-time.

## Features

- Recognition of handwritten mathematical symbols (digits, operators, parentheses, etc.)
- Interactive GUI for drawing and recognizing characters
- Pre-trained CNN model for immediate use
- Support for training custom models with your own data
- Evaluation tools to measure model accuracy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/downquark7/MISS.git
   cd MISS
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Interactive GUI

To use the interactive drawing interface for character recognition:

```
python model_gui.py
```

This will open a window where you can:
- Draw a mathematical symbol using your mouse
- Click "Predict" to recognize the drawn character
- Click "Clear" to reset the canvas

### Training a New Model

To train a new model with your own data:

```
python tensor_flow_model.py
```

You can modify the training parameters in the script as needed.

### Evaluating Model Performance

To evaluate the model on test images:

```
python tensor_flow_model.py
```

The script will run evaluation on the test images and display accuracy metrics.

## Project Structure

- `base_model.py`: Base class for character recognition models
- `tensor_flow_model.py`: TensorFlow CNN implementation for character recognition
- `model_gui.py`: Interactive GUI for drawing and recognizing characters
- `data_labeling_ui.py`: Tool for labeling new training data
- `add_new_data.py`: Script for adding new training data
- `character_model.keras`: Pre-trained model file
- `label_mappings.npy`: Mapping between indices and character labels
- `labels.csv`: Dataset labels
- `requirements.txt`: Required Python packages

## Dependencies

- numpy
- pandas
- opencv-python
- scikit-learn
- tensorflow
- matplotlib
- pillow

## Dataset

The project uses the Handwritten Math Expressions Dataset from Kaggle:
https://www.kaggle.com/datasets/govindaramsriram/handwritten-math-expressions-dataset

## Acknowledgments

This project was started based on the Kaggle notebook:
https://www.kaggle.com/code/aruneembhowmick/processing-images-of-arithmetic-expressions

## License

This project is licensed under the Creative Commons Attribution 4.0 International License:
https://creativecommons.org/licenses/by/4.0/