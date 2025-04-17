import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import original classifiers
from KNN import KNN
from pca_classifier import PCAClassifier
from tensor_flow_model import TensorFlowModel

# Import resized classifiers
from resized_knn import ResizedKNN
from resized_pca_classifier import ResizedPCAClassifier
from resized_tensor_flow_model import ResizedTensorFlowModel

def evaluate_classifier(classifier, name, test_folder, expected_chars):
    """Evaluate a classifier and return its accuracy, training time, and inference time."""
    print(f"\n{'-'*20} Evaluating {name} {'-'*20}")

    # Measure training time
    start_time = time.time()
    if hasattr(classifier, 'train_model'):
        # For classifiers that don't train in __init__
        if name.startswith('TensorFlow'):
            # TensorFlow models have a different interface
            classifier.train_model()
        else:
            # Other models train in __init__, so we don't need to call train_model again
            pass
    training_time = time.time() - start_time

    # Measure inference time and accuracy
    start_time = time.time()
    accuracy = classifier.eval_folder(test_folder, expected_chars, plot=False)
    inference_time = time.time() - start_time

    return {
        'name': name,
        'accuracy': accuracy,
        'training_time': training_time,
        'inference_time': inference_time
    }

def main():
    test_folder = '0_)_test_images'
    expected_chars = '0123456789+*/=()'

    # List of classifiers to evaluate
    classifiers = [
        (KNN(), "KNN (28x28)"),
        (ResizedKNN(), "KNN (14x14)"),
        (PCAClassifier(), "PCA (28x28)"),
        (ResizedPCAClassifier(), "PCA (14x14)"),
        (TensorFlowModel(train=True), "TensorFlow (28x28)"),
        (ResizedTensorFlowModel(train=True), "TensorFlow (14x14)")
    ]

    # Evaluate each classifier
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
            f"{result['training_time']:.2f}s",
            f"{result['inference_time']:.2f}s"
        ])

    # Print table header
    print(f"{'Classifier':<20} {'Accuracy':<15} {'Training Time':<15} {'Inference Time':<15}")
    print("-" * 65)

    # Print table rows
    for row in table_data:
        print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15}")

    # Group results by classifier type
    knn_results = [r for r in results if "KNN" in r['name']]
    pca_results = [r for r in results if "PCA" in r['name']]
    tf_results = [r for r in results if "TensorFlow" in r['name']]

    # Print comparison for each classifier type
    for classifier_type, type_results in [("KNN", knn_results), ("PCA", pca_results), ("TensorFlow", tf_results)]:
        if len(type_results) == 2:
            original = type_results[0]
            resized = type_results[1]

            accuracy_diff = resized['accuracy'] - original['accuracy']
            training_time_diff = resized['training_time'] - original['training_time']
            inference_time_diff = resized['inference_time'] - original['inference_time']

            print(f"\n{classifier_type} Comparison:")
            print(f"  Accuracy: {accuracy_diff:.2f}% ({'better' if accuracy_diff > 0 else 'worse'})")
            print(f"  Training Time: {training_time_diff:.2f}s ({'slower' if training_time_diff > 0 else 'faster'})")
            print(f"  Inference Time: {inference_time_diff:.2f}s ({'slower' if inference_time_diff > 0 else 'faster'})")

    # Create bar charts for visualization
    create_comparison_charts(results)

def create_comparison_charts(results):
    """Create bar charts comparing original and resized classifiers."""
    # Extract data for plotting
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    training_times = [r['training_time'] for r in results]
    inference_times = [r['inference_time'] for r in results]

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Accuracy comparison
    ax1.bar(names, accuracies)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 1, f"{v:.2f}%", ha='center')

    # Training time comparison
    ax2.bar(names, training_times)
    ax2.set_title('Training Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    for i, v in enumerate(training_times):
        ax2.text(i, v + 0.1, f"{v:.2f}s", ha='center')

    # Inference time comparison
    ax3.bar(names, inference_times)
    ax3.set_title('Inference Time Comparison')
    ax3.set_ylabel('Time (seconds)')
    for i, v in enumerate(inference_times):
        ax3.text(i, v + 0.1, f"{v:.2f}s", ha='center')

    plt.tight_layout()
    plt.savefig('classifier_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
