import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import original classifiers
from KNN import KNN
from pca_classifier import PCAClassifier
from tensor_flow_model import TensorFlowModel
from improved_tensor_flow_model import ImprovedTensorFlowModel
from improved_tensor_flow_model_v2 import ImprovedTensorFlowModelV2
from decision_tree_classifier import DecisionTreeModel
from gaussian_naive_bayes_classifier import GaussianNaiveBayesModel
from gradient_boosting_classifier import GradientBoostingModel
from logistic_regression_classifier import LogisticRegressionModel
from neural_network_classifier import NeuralNetworkModel
from random_forest_model import RandomForestModel
from svm_model import SVMModel
from meta_classifier import MetaClassifier
from improved_meta_classifier import ImprovedMetaClassifier
from transfer_learning_model import TransferLearningModel

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
        # For classifiers that need explicit training
        if any(x in name for x in ['TensorFlow', 'ImprovedTensorFlow', 'TransferLearning']):
            # Models that need explicit training
            classifier.train_model()
        else:
            # Other models might train in __init__, so we don't need to call train_model again
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
        (ResizedTensorFlowModel(train=True), "TensorFlow (14x14)"),
        (ImprovedTensorFlowModel(train=True), "ImprovedTensorFlow"),
        (ImprovedTensorFlowModelV2(train=True), "ImprovedTensorFlowV2"),
        (DecisionTreeModel(), "DecisionTree"),
        (GaussianNaiveBayesModel(), "GaussianNaiveBayes"),
        (GradientBoostingModel(), "GradientBoosting"),
        (LogisticRegressionModel(), "LogisticRegression"),
        (NeuralNetworkModel(), "NeuralNetwork"),
        (RandomForestModel(), "RandomForest"),
        (SVMModel(), "SVM"),
        # Skipping the following classifiers as they take too long to run:
        # (TransferLearningModel(train=True), "TransferLearning"),
        # (MetaClassifier(), "MetaClassifier"),
        # (ImprovedMetaClassifier(), "ImprovedMetaClassifier")
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
    tf_results = [r for r in results if "TensorFlow" in r['name'] and not "Improved" in r['name']]
    improved_tf_results = [r for r in results if "ImprovedTensorFlow" in r['name']]
    meta_results = [r for r in results if "MetaClassifier" in r['name']]

    # Other individual classifiers
    other_classifiers = [
        r for r in results if not any(x in r['name'] for x in 
        ["KNN", "PCA", "TensorFlow", "ImprovedTensorFlow", "MetaClassifier"])
    ]

    print("\n" + "="*50)
    print("CLASSIFIER TYPE COMPARISONS")
    print("="*50)

    # Compare original vs resized for KNN, PCA, TensorFlow
    for classifier_type, type_results in [("KNN", knn_results), ("PCA", pca_results), ("TensorFlow", tf_results)]:
        if len(type_results) == 2:
            original = type_results[0]
            resized = type_results[1]

            accuracy_diff = resized['accuracy'] - original['accuracy']
            training_time_diff = resized['training_time'] - original['training_time']
            inference_time_diff = resized['inference_time'] - original['inference_time']

            print(f"\n{classifier_type} Original vs Resized Comparison:")
            print(f"  Accuracy: {accuracy_diff:.2f}% ({'better' if accuracy_diff > 0 else 'worse'})")
            print(f"  Training Time: {training_time_diff:.2f}s ({'slower' if training_time_diff > 0 else 'faster'})")
            print(f"  Inference Time: {inference_time_diff:.2f}s ({'slower' if inference_time_diff > 0 else 'faster'})")

    # Compare TensorFlow vs ImprovedTensorFlow models
    if tf_results and improved_tf_results:
        original_tf = next((r for r in tf_results if "(28x28)" in r['name']), None)
        if original_tf:
            print("\nTensorFlow vs Improved Models Comparison:")
            for improved in improved_tf_results:
                accuracy_diff = improved['accuracy'] - original_tf['accuracy']
                training_time_diff = improved['training_time'] - original_tf['training_time']
                inference_time_diff = improved['inference_time'] - original_tf['inference_time']

                print(f"\n  {improved['name']} vs {original_tf['name']}:")
                print(f"    Accuracy: {accuracy_diff:.2f}% ({'better' if accuracy_diff > 0 else 'worse'})")
                print(f"    Training Time: {training_time_diff:.2f}s ({'slower' if training_time_diff > 0 else 'faster'})")
                print(f"    Inference Time: {inference_time_diff:.2f}s ({'slower' if inference_time_diff > 0 else 'faster'})")

    # Compare MetaClassifier vs ImprovedMetaClassifier
    if len(meta_results) == 2:
        meta = next((r for r in meta_results if r['name'] == "MetaClassifier"), None)
        improved_meta = next((r for r in meta_results if r['name'] == "ImprovedMetaClassifier"), None)

        if meta and improved_meta:
            accuracy_diff = improved_meta['accuracy'] - meta['accuracy']
            training_time_diff = improved_meta['training_time'] - meta['training_time']
            inference_time_diff = improved_meta['inference_time'] - meta['inference_time']

            print("\nMetaClassifier vs ImprovedMetaClassifier Comparison:")
            print(f"  Accuracy: {accuracy_diff:.2f}% ({'better' if accuracy_diff > 0 else 'worse'})")
            print(f"  Training Time: {training_time_diff:.2f}s ({'slower' if training_time_diff > 0 else 'faster'})")
            print(f"  Inference Time: {inference_time_diff:.2f}s ({'slower' if inference_time_diff > 0 else 'faster'})")

    # Create bar charts for visualization
    create_comparison_charts(results)

def create_comparison_charts(results):
    """Create bar charts comparing all classifiers."""
    # Extract data for plotting
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    training_times = [r['training_time'] for r in results]
    inference_times = [r['inference_time'] for r in results]

    # Determine figure size based on number of classifiers
    fig_width = max(10, len(names) * 0.8)
    fig_height = 18

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width, fig_height))

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

    # Training time comparison
    bars2 = ax2.bar(x_pos, training_times)
    ax2.set_title('Training Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.2f}s", ha='center', va='bottom')

    # Inference time comparison
    bars3 = ax3.bar(x_pos, inference_times)
    ax3.set_title('Inference Time Comparison')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.2f}s", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('classifier_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
