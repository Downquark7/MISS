import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import available classifiers
from KNN import KNN
from naive_KNN import NaiveKNN
from CNN import CNN

def evaluate_classifier(classifier, name, test_folder, expected_chars):
    """Evaluate a classifier and return its accuracy, per-character accuracy, and inference time."""
    print(f"\n{'-'*20} Evaluating {name} {'-'*20}")

    # Train the model if needed
    if hasattr(classifier, 'train_model'):
        # For classifiers that need explicit training
        if any(x in name for x in ['TensorFlow']):
            # Models that need explicit training
            classifier.train_model()
        else:
            # Other models might train in __init__, so we don't need to call train_model again
            pass

    # Measure inference time and accuracy
    start_time = time.time()

    # Calculate per-character accuracy
    file_paths = [os.path.join(test_folder, f) for f in os.listdir(test_folder)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    total_chars = 0
    correct_matches = 0
    char_accuracy = {char: {"total": 0, "correct": 0} for char in set(expected_chars)}

    for img_path in file_paths:
        read_chars = classifier.scan_img_path(img_path, plot=False)
        for rc, cl in zip(read_chars, expected_chars):
            total_chars += 1
            if rc == cl:
                correct_matches += 1
                char_accuracy[cl]["correct"] += 1
            char_accuracy[cl]["total"] += 1

    overall_accuracy = (correct_matches / total_chars) * 100 if total_chars > 0 else 0
    per_char_accuracy = {char: (data["correct"] / data["total"] * 100 if data["total"] > 0 else 0)
                         for char, data in char_accuracy.items()}

    inference_time = time.time() - start_time

    return {
        'name': name,
        'accuracy': overall_accuracy,
        'per_char_accuracy': per_char_accuracy,
        'inference_time': inference_time
    }

def main():
    # Run the actual evaluation of classifiers
    test_folder = '0_)_test_images'
    expected_chars = '0123456789+*/=()'
    classifiers = [
        (KNN(), "KNN"),
        (NaiveKNN(), "NaiveKNN"),
        (CNN(train=False), "TensorFlow")
    ]
    results = []
    for classifier, name in classifiers:
        result = evaluate_classifier(classifier, name, test_folder, expected_chars)
        results.append(result)

    # Print results as a table
    print("\n" + "="*50)
    print("CLASSIFIER COMPARISON RESULTS")
    print("="*50)

    table_data = []
    for result in results:
        table_data.append([
            result['name'],
            f"{result['accuracy']:.2f}%",
            f"{result['inference_time']:.2f}s"
        ])

    # Print table header
    print(f"{'Classifier':<20} {'Accuracy':<15} {'Inference Time':<15}")
    print("-" * 50)

    # Print table rows
    for row in table_data:
        print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15}")

    # Create bar charts for visualization
    create_comparison_charts(results)

    # Create character accuracy charts
    create_character_accuracy_chart(results)

def create_comparison_charts(results):
    """Create bar charts comparing all classifiers."""
    # Extract data for plotting
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    inference_times = [r['inference_time'] for r in results]

    # Determine figure size based on number of classifiers
    fig_width = max(10, len(names) * 0.8)
    fig_height = 12

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))

    # Create x positions for bars
    x_pos = np.arange(len(names))

    # Accuracy comparison
    bars1 = ax1.bar(x_pos, accuracies)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{height:.2f}%", ha='center', va='bottom')

    # Inference time comparison
    bars2 = ax2.bar(x_pos, inference_times)
    ax2.set_title('Inference Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.2f}s", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('classifier_comparison.png')
    plt.show()

def create_character_accuracy_chart(results):
    """Create a heatmap showing which models perform best for each character."""
    # Extract data for plotting
    names = [r['name'] for r in results]

    # Get all unique characters across all models
    all_chars = set()
    for result in results:
        all_chars.update(result['per_char_accuracy'].keys())
    all_chars = sorted(list(all_chars))

    # Create a matrix of accuracies: rows=models, columns=characters
    accuracy_matrix = np.zeros((len(names), len(all_chars)))
    for i, result in enumerate(results):
        for j, char in enumerate(all_chars):
            if char in result['per_char_accuracy']:
                accuracy_matrix[i, j] = result['per_char_accuracy'][char]

    # Create a heatmap
    plt.figure(figsize=(max(12, len(all_chars) * 0.8), max(8, len(names) * 0.5)))

    # Create the heatmap
    im = plt.imshow(accuracy_matrix, cmap='viridis', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Accuracy (%)')

    # Add labels
    plt.yticks(np.arange(len(names)), names)
    plt.xticks(np.arange(len(all_chars)), all_chars)
    plt.xlabel('Character')
    plt.ylabel('Model')
    plt.title('Model Accuracy by Character')

    # Add text annotations in the cells
    for i in range(len(names)):
        for j in range(len(all_chars)):
            text = plt.text(j, i, f"{accuracy_matrix[i, j]:.1f}%",
                           ha="center", va="center", color="w" if accuracy_matrix[i, j] < 70 else "black")

    plt.tight_layout()
    plt.savefig('character_accuracy_heatmap.png')
    plt.show()

    # Create a bar chart showing the best model for each character
    plt.figure(figsize=(max(10, len(all_chars) * 0.8), 6))

    # Find the best model for each character
    best_models = []
    best_accuracies = []

    for j, char in enumerate(all_chars):
        best_model_idx = np.argmax(accuracy_matrix[:, j])
        best_models.append(names[best_model_idx])
        best_accuracies.append(accuracy_matrix[best_model_idx, j])

    # Create a bar chart
    bars = plt.bar(all_chars, best_accuracies)

    # Add model names as text on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{best_models[i]}", ha='center', va='bottom', rotation=90, fontsize=8)

    plt.ylim(0, 105)  # Leave room for model names
    plt.xlabel('Character')
    plt.ylabel('Best Accuracy (%)')
    plt.title('Best Model Accuracy by Character')

    plt.tight_layout()
    plt.savefig('best_model_by_character.png')
    plt.show()

if __name__ == "__main__":
    main()
