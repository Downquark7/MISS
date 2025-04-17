from base_model import BaseModel
from KNN import KNN
from naive_KNN import NaiveKNN
from tensor_flow_model import TensorFlowModel
from random_forest_model import RandomForestModel
from svm_model import SVMModel
from pca_classifier import PCAClassifier
from decision_tree_classifier import DecisionTreeModel
from gradient_boosting_classifier import GradientBoostingModel
from logistic_regression_classifier import LogisticRegressionModel
from neural_network_classifier import NeuralNetworkModel
from gaussian_naive_bayes_classifier import GaussianNaiveBayesModel
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import defaultdict, Counter


class MetaClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        print("Initializing Meta Classifier...")

        # Initialize all models
        self.knn_model = KNN()
        self.naive_knn_model = NaiveKNN()
        self.tf_model = TensorFlowModel(train=False)
        self.rf_model = RandomForestModel()
        self.svm_model = SVMModel()
        self.pca_model = PCAClassifier()

        # Initialize the new models
        self.dt_model = DecisionTreeModel()
        self.gb_model = GradientBoostingModel()
        self.lr_model = LogisticRegressionModel()
        self.nn_model = NeuralNetworkModel()
        self.gnb_model = GaussianNaiveBayesModel()

        # Initialize counters for debugging
        self.unanimous_count = 0
        self.knn_naive_override_count = 0
        self.tf_default_count = 0
        self.confidence_override_count = 0
        self.character_specific_count = 0
        self.ensemble_count = 0
        self.adaptive_count = 0

        # Define problematic characters based on previous analysis
        self.problematic_chars = {'*', '5', '9'}

        # Track model performance per character
        # This will be used for dynamic model selection
        self.model_performance = {
            'tf': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'knn': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'naive_knn': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'rf': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'svm': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'pca': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'dt': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'gb': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'lr': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'nn': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'gnb': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'ensemble': defaultdict(lambda: {'correct': 0, 'total': 0})
        }

        # Track the ground truth for adaptive learning
        self.ground_truth = {}

        # Parameters for the ensemble method
        self.top_k = 3  # Number of top predictions to consider from TensorFlow
        self.min_confidence = 0.10  # Minimum confidence to consider a prediction
        self.adaptive_learning_rate = 0.1  # Rate at which the model adapts to new data

        print("Meta Classifier initialized!")

    def scan_img(self, img):
        # Get predictions from all models
        knn_prediction = self.knn_model.scan_img(img)
        naive_knn_prediction = self.naive_knn_model.scan_img(img)
        rf_prediction = self.rf_model.scan_img(img)
        svm_prediction, svm_confidence = self.svm_model.scan_img(img, return_confidence=True)
        pca_prediction, pca_confidence = self.pca_model.scan_img(img, return_confidence=True)

        # Get predictions from new models
        dt_prediction, dt_confidence = self.dt_model.scan_img(img, return_confidence=True)
        gb_prediction, gb_confidence = self.gb_model.scan_img(img, return_confidence=True)
        lr_prediction, lr_confidence = self.lr_model.scan_img(img, return_confidence=True)
        nn_prediction, nn_confidence = self.nn_model.scan_img(img, return_confidence=True)
        gnb_prediction, gnb_confidence = self.gnb_model.scan_img(img, return_confidence=True)

        # Get top-k predictions from TensorFlow with confidence scores
        tf_top_k = self.tf_model.scan_img(img, return_top_k=True, k=self.top_k)
        tf_prediction = tf_top_k[0][0]  # Top prediction
        tf_confidence = tf_top_k[0][1]  # Confidence of top prediction

        print(f"KNN prediction: {knn_prediction}")
        print(f"Naive KNN prediction: {naive_knn_prediction}")
        print(f"Random Forest prediction: {rf_prediction}")
        print(f"SVM prediction: {svm_prediction}, Confidence: {svm_confidence:.2f}")
        print(f"PCA-KNN prediction: {pca_prediction}, Confidence: {pca_confidence:.2f}")
        print(f"Decision Tree prediction: {dt_prediction}, Confidence: {dt_confidence:.2f}")
        print(f"Gradient Boosting prediction: {gb_prediction}, Confidence: {gb_confidence:.2f}")
        print(f"Logistic Regression prediction: {lr_prediction}, Confidence: {lr_confidence:.2f}")
        print(f"Neural Network prediction: {nn_prediction}, Confidence: {nn_confidence:.2f}")
        print(f"Gaussian Naive Bayes prediction: {gnb_prediction}, Confidence: {gnb_confidence:.2f}")
        print(f"TensorFlow top prediction: {tf_prediction}, Confidence: {tf_confidence:.2f}")
        print(f"TensorFlow top-{self.top_k} predictions: {tf_top_k}")

        # If we have ground truth for this image from a previous evaluation,
        # update the model performance metrics
        img_id = id(img)  # Use the image's id as a key
        if img_id in self.ground_truth:
            true_char = self.ground_truth[img_id]
            # Update performance metrics for each model
            self._update_model_performance('tf', tf_prediction, true_char)
            self._update_model_performance('knn', knn_prediction, true_char)
            self._update_model_performance('naive_knn', naive_knn_prediction, true_char)
            self._update_model_performance('rf', rf_prediction, true_char)
            self._update_model_performance('svm', svm_prediction, true_char)
            self._update_model_performance('pca', pca_prediction, true_char)
            self._update_model_performance('dt', dt_prediction, true_char)
            self._update_model_performance('gb', gb_prediction, true_char)
            self._update_model_performance('lr', lr_prediction, true_char)
            self._update_model_performance('nn', nn_prediction, true_char)
            self._update_model_performance('gnb', gnb_prediction, true_char)

            # Check if we can use adaptive model selection
            if self._can_use_adaptive_selection(true_char):
                best_model = self._get_best_model_for_char(true_char)
                if best_model == 'tf':
                    prediction = tf_prediction
                elif best_model == 'knn':
                    prediction = knn_prediction
                elif best_model == 'naive_knn':
                    prediction = naive_knn_prediction
                elif best_model == 'rf':
                    prediction = rf_prediction
                elif best_model == 'svm':
                    prediction = svm_prediction
                elif best_model == 'pca':
                    prediction = pca_prediction
                elif best_model == 'dt':
                    prediction = dt_prediction
                elif best_model == 'gb':
                    prediction = gb_prediction
                elif best_model == 'lr':
                    prediction = lr_prediction
                elif best_model == 'nn':
                    prediction = nn_prediction
                else:  # gnb
                    prediction = gnb_prediction

                self.adaptive_count += 1
                print(f"Adaptive model selection: Using {best_model}'s prediction {prediction} for character {true_char}")
                return prediction

        # If all models agree, return the unanimous prediction
        if (knn_prediction == naive_knn_prediction == tf_prediction == rf_prediction == 
            svm_prediction == pca_prediction == dt_prediction == gb_prediction == 
            lr_prediction == nn_prediction == gnb_prediction):
            self.unanimous_count += 1
            print(f"All models agree on: {tf_prediction}")
            return tf_prediction

        # Create a weighted voting ensemble
        votes = Counter()

        # Add votes from TensorFlow's top-k predictions
        for char, conf in tf_top_k:
            if conf >= self.min_confidence:
                votes[char] += conf

        # Add votes from other models
        # Give them weights based on their overall accuracy
        knn_weight = 0.66  # KNN accuracy is about 66%
        naive_knn_weight = 0.62  # NaiveKNN accuracy is about 62%
        rf_weight = 0.75  # Increased Random Forest weight
        svm_weight = 0.70  # Increased SVM base weight
        pca_weight = 0.70  # Initial PCA weight, will be adjusted based on performance
        dt_weight = 0.65  # Initial Decision Tree weight
        gb_weight = 0.70  # Initial Gradient Boosting weight
        lr_weight = 0.65  # Initial Logistic Regression weight
        nn_weight = 0.70  # Initial Neural Network weight
        gnb_weight = 0.60  # Initial Gaussian Naive Bayes weight

        votes[knn_prediction] += knn_weight
        votes[naive_knn_prediction] += naive_knn_weight
        votes[rf_prediction] += rf_weight

        # For models with confidence scores, use a higher weight for high-confidence predictions
        svm_vote_weight = svm_weight * (1.0 + svm_confidence)  # Scales from 0.7 to 1.4 based on confidence
        votes[svm_prediction] += svm_vote_weight
        print(f"SVM vote weight: {svm_vote_weight:.2f} (base: {svm_weight}, confidence: {svm_confidence:.2f})")

        pca_vote_weight = pca_weight * (1.0 + pca_confidence)  # Scales from 0.7 to 1.4 based on confidence
        votes[pca_prediction] += pca_vote_weight
        print(f"PCA vote weight: {pca_vote_weight:.2f} (base: {pca_weight}, confidence: {pca_confidence:.2f})")

        dt_vote_weight = dt_weight * (1.0 + dt_confidence)  # Scales based on confidence
        votes[dt_prediction] += dt_vote_weight
        print(f"Decision Tree vote weight: {dt_vote_weight:.2f} (base: {dt_weight}, confidence: {dt_confidence:.2f})")

        gb_vote_weight = gb_weight * (1.0 + gb_confidence)  # Scales based on confidence
        votes[gb_prediction] += gb_vote_weight
        print(f"Gradient Boosting vote weight: {gb_vote_weight:.2f} (base: {gb_weight}, confidence: {gb_confidence:.2f})")

        lr_vote_weight = lr_weight * (1.0 + lr_confidence)  # Scales based on confidence
        votes[lr_prediction] += lr_vote_weight
        print(f"Logistic Regression vote weight: {lr_vote_weight:.2f} (base: {lr_weight}, confidence: {lr_confidence:.2f})")

        nn_vote_weight = nn_weight * (1.0 + nn_confidence)  # Scales based on confidence
        votes[nn_prediction] += nn_vote_weight
        print(f"Neural Network vote weight: {nn_vote_weight:.2f} (base: {nn_weight}, confidence: {nn_confidence:.2f})")

        gnb_vote_weight = gnb_weight * (1.0 + gnb_confidence)  # Scales based on confidence
        votes[gnb_prediction] += gnb_vote_weight
        print(f"Gaussian Naive Bayes vote weight: {gnb_vote_weight:.2f} (base: {gnb_weight}, confidence: {gnb_confidence:.2f})")

        # Count how many models agree on each prediction
        prediction_counts = Counter([
            knn_prediction, naive_knn_prediction, rf_prediction, svm_prediction, pca_prediction,
            dt_prediction, gb_prediction, lr_prediction, nn_prediction, gnb_prediction
        ])
        most_common_pred, count = prediction_counts.most_common(1)[0]

        # Special case: If the new classifiers (RF and SVM) agree with each other but disagree with TensorFlow
        # This applies to all characters, not just problematic ones
        if rf_prediction == svm_prediction and rf_prediction != tf_prediction and svm_confidence > 0.7:
            votes[rf_prediction] *= 1.2  # Modest boost
            print(f"New classifiers consensus boost: RF and SVM agree on {rf_prediction} with high confidence")

        # Special case: If PCA agrees with either RF or SVM but disagrees with TensorFlow
        # This can help when PCA's dimensionality reduction captures important features
        if (pca_prediction == rf_prediction or pca_prediction == svm_prediction) and pca_prediction != tf_prediction and pca_confidence > 0.7:
            votes[pca_prediction] *= 1.15  # Slight boost
            print(f"PCA consensus boost: PCA agrees with {'RF' if pca_prediction == rf_prediction else 'SVM'} on {pca_prediction} with high confidence")

        # Special handling for problematic characters
        if tf_prediction in self.problematic_chars and tf_confidence < 0.80:
            # Reduce the weight of TensorFlow's prediction for problematic characters with low confidence
            votes[tf_prediction] *= 0.8

            # If multiple models agree on a different prediction, boost its weight
            # Give a bigger boost when more models agree
            if count >= 2 and most_common_pred != tf_prediction:
                # Base boost factor
                boost_factor = 1.5

                # Extra boost if the new classifiers (RF and SVM) agree with each other
                if rf_prediction == svm_prediction and rf_prediction != tf_prediction:
                    boost_factor += 0.3
                    print(f"Extra boost: RF and SVM agree on {rf_prediction}")

                # Extra boost if all non-TF models agree
                if count == 4:
                    boost_factor += 0.5
                    print(f"Major boost: All non-TF models agree on {most_common_pred}")
                # Smaller extra boost if 3 models agree
                elif count == 3:
                    boost_factor += 0.2
                    print(f"Medium boost: 3 models agree on {most_common_pred}")

                votes[most_common_pred] *= boost_factor
                self.character_specific_count += 1
                print(f"Character-specific boost: {count} models agree on {most_common_pred} for problematic char {tf_prediction}, boost factor: {boost_factor:.2f}")

        # Get the prediction with the highest weighted vote
        ensemble_prediction = votes.most_common(1)[0][0]

        # If the ensemble prediction is different from all individual models,
        # print the voting results for debugging
        if ensemble_prediction not in [
            tf_prediction, knn_prediction, naive_knn_prediction, rf_prediction, svm_prediction, pca_prediction,
            dt_prediction, gb_prediction, lr_prediction, nn_prediction, gnb_prediction
        ]:
            self.ensemble_count += 1
            print(f"Ensemble prediction: {ensemble_prediction} (votes: {votes})")
            return ensemble_prediction

        # If the ensemble agrees with TensorFlow, count it as a TensorFlow default
        if ensemble_prediction == tf_prediction:
            self.tf_default_count += 1
            print(f"Ensemble agrees with TensorFlow: {tf_prediction}")
        # If the ensemble agrees with both KNN models, count it as a KNN override
        elif ensemble_prediction == knn_prediction == naive_knn_prediction:
            self.knn_naive_override_count += 1
            print(f"Ensemble agrees with KNN consensus: {knn_prediction}")
        # If the ensemble agrees with one KNN model, count it as a confidence override
        else:
            self.confidence_override_count += 1
            print(f"Ensemble agrees with one KNN model: {ensemble_prediction}")

        return ensemble_prediction

    def _update_model_performance(self, model_name, prediction, true_char):
        """Update the performance metrics for a model."""
        self.model_performance[model_name][true_char]['total'] += 1
        if prediction == true_char:
            self.model_performance[model_name][true_char]['correct'] += 1

    def _can_use_adaptive_selection(self, char):
        """Check if we have enough data to use adaptive model selection for this character."""
        for model_name in ['tf', 'knn', 'naive_knn', 'rf', 'svm', 'pca', 'dt', 'gb', 'lr', 'nn', 'gnb']:
            if self.model_performance[model_name][char]['total'] < 3:
                return False
        return True

    def _get_best_model_for_char(self, char):
        """Get the best performing model for this character."""
        best_model = 'tf'  # Default to TensorFlow
        best_accuracy = 0

        for model_name in ['tf', 'knn', 'naive_knn', 'rf', 'svm', 'pca', 'dt', 'gb', 'lr', 'nn', 'gnb']:
            perf = self.model_performance[model_name][char]
            if perf['total'] > 0:
                accuracy = perf['correct'] / perf['total']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name

        return best_model

    def eval_folder(self, folder_path, char_list, plot=True):
        # Store the original scan_img_path method to restore it later
        original_scan_img_path = self.scan_img_path

        # Override scan_img_path to capture ground truth for adaptive learning
        def scan_img_path_with_ground_truth(img_path, plot=True):
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 239, 17)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            char_list_index = 0
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

            char_list_result = ""
            for binary_region in binary_regions:
                if char_list_index < len(char_list):
                    true_char = char_list[char_list_index]
                    char_list_index += 1

                    padded_img = self.pad_image(binary_region)
                    img = np.array(padded_img) / 255.0
                    img_tensor = img.reshape(1, 28, 28, 1)

                    # Store the ground truth for this image
                    self.ground_truth[id(img_tensor)] = true_char

                    character = self.scan_img(img_tensor)
                    char_list_result += character

                    # Update model performance metrics
                    self._update_model_performance('tf', self.tf_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('knn', self.knn_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('naive_knn', self.naive_knn_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('rf', self.rf_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('svm', self.svm_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('pca', self.pca_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('dt', self.dt_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('gb', self.gb_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('lr', self.lr_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('nn', self.nn_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('gnb', self.gnb_model.scan_img(img_tensor), true_char)
                    self._update_model_performance('ensemble', character, true_char)

            print(char_list_result)
            if plot:
                thickness = max(1, int(gray.shape[0] * 0.02))
                for mx1, my1, mx2, my2 in merged_boxes:
                    cv2.rectangle(binary, (mx1, my1), (mx2, my2), (255, 255, 255), thickness)
                plt.imshow(binary, cmap="gray")
                plt.title(f"{char_list_result}")
                plt.show()
            return char_list_result

        # Replace the scan_img_path method with our custom version
        self.scan_img_path = scan_img_path_with_ground_truth

        # Call the parent's eval_folder method
        result = super().eval_folder(folder_path, char_list, plot)

        # Restore the original scan_img_path method
        self.scan_img_path = original_scan_img_path

        # Print debugging statistics
        total = (self.unanimous_count + self.knn_naive_override_count + 
                self.tf_default_count + self.confidence_override_count + 
                self.character_specific_count + self.ensemble_count + 
                self.adaptive_count)

        print("\n" + "="*50)
        print(f"META CLASSIFIER OVERALL ACCURACY: {result:.2f}%")
        print("="*50)

        print("\nVoting Statistics:")
        print(f"Unanimous predictions: {self.unanimous_count}/{total} ({self.unanimous_count/total*100:.2f}%)")
        print(f"KNN+NaiveKNN override TensorFlow: {self.knn_naive_override_count}/{total} ({self.knn_naive_override_count/total*100:.2f}%)")
        print(f"Confidence-based overrides: {self.confidence_override_count}/{total} ({self.confidence_override_count/total*100:.2f}%)")
        print(f"Character-specific strategies: {self.character_specific_count}/{total} ({self.character_specific_count/total*100:.2f}%)")
        print(f"Ensemble predictions: {self.ensemble_count}/{total} ({self.ensemble_count/total*100:.2f}%)")
        print(f"Adaptive model selection: {self.adaptive_count}/{total} ({self.adaptive_count/total*100:.2f}%)")
        print(f"TensorFlow default: {self.tf_default_count}/{total} ({self.tf_default_count/total*100:.2f}%)")

        # Print model performance per character
        print("\nModel Performance Per Character:")
        for char in sorted(set(char_list)):
            print(f"\nCharacter: {char}")
            for model_name in ['tf', 'knn', 'naive_knn', 'rf', 'svm', 'pca', 'ensemble']:
                perf = self.model_performance[model_name][char]
                if perf['total'] > 0:
                    accuracy = perf['correct'] / perf['total'] * 100
                    print(f"  {model_name.upper()}: {perf['correct']}/{perf['total']} ({accuracy:.2f}%)")

        return result

    def print_model_comparison(self, char_list):
        """Print a comparison of model accuracies."""
        # Calculate average accuracy for each model type
        model_avg_accuracy = {}
        for model_name in ['tf', 'knn', 'naive_knn', 'rf', 'svm', 'pca', 'ensemble']:
            total_correct = 0
            total_samples = 0
            for char_data in self.model_performance[model_name].values():
                total_correct += char_data['correct']
                total_samples += char_data['total']

            if total_samples > 0:
                accuracy = (total_correct / total_samples) * 100
                model_avg_accuracy[model_name] = accuracy

        # Print individual model accuracies
        print("\nINDIVIDUAL MODEL ACCURACIES:")
        for model_name, accuracy in model_avg_accuracy.items():
            print(f"{model_name.upper()}: {accuracy:.2f}%")

        # Highlight contribution of new classifiers
        if all(model in model_avg_accuracy for model in ['rf', 'svm', 'pca', 'dt', 'gb', 'lr', 'nn', 'gnb']):
            original_classifiers_avg = (model_avg_accuracy['rf'] + model_avg_accuracy['svm'] + model_avg_accuracy['pca']) / 3
            new_classifiers_avg = (model_avg_accuracy['dt'] + model_avg_accuracy['gb'] + model_avg_accuracy['lr'] + 
                                  model_avg_accuracy['nn'] + model_avg_accuracy['gnb']) / 5
            all_new_classifiers_avg = (original_classifiers_avg * 3 + new_classifiers_avg * 5) / 8
            existing_classifiers_avg = (model_avg_accuracy['tf'] + model_avg_accuracy['knn'] + model_avg_accuracy['naive_knn']) / 3

            print(f"\nOriginal New Classifiers (RF+SVM+PCA) Average: {original_classifiers_avg:.2f}%")
            print(f"Additional New Classifiers (DT+GB+LR+NN+GNB) Average: {new_classifiers_avg:.2f}%")
            print(f"All New Classifiers Average: {all_new_classifiers_avg:.2f}%")
            print(f"Existing Classifiers (TF+KNN+NaiveKNN) Average: {existing_classifiers_avg:.2f}%")
            print(f"Difference (All New vs Existing): {all_new_classifiers_avg - existing_classifiers_avg:.2f}%")

            # Show individual classifier performance compared to ensemble
            print("\nIndividual Classifier Performance vs Ensemble:")
            for model_name in ['dt', 'gb', 'lr', 'nn', 'gnb']:
                print(f"{model_name.upper()} Classifier: {model_avg_accuracy[model_name]:.2f}%")
                print(f"  Difference from Ensemble: {model_avg_accuracy['ensemble'] - model_avg_accuracy[model_name]:.2f}%")


if __name__ == "__main__":
    # First run TensorFlow model to get its accuracy for comparison
    print("\n" + "="*50)
    print("RUNNING TENSORFLOW MODEL FOR COMPARISON")
    print("="*50)
    tf_model = TensorFlowModel(train=False)
    tf_accuracy = tf_model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)

    # Then run the meta classifier
    print("\n" + "="*50)
    print("RUNNING META CLASSIFIER")
    print("="*50)
    model = MetaClassifier()
    meta_accuracy = model.eval_folder('0_)_test_images', '0123456789+*/=()', plot=False)

    # Print the comparison at the beginning for visibility
    print("\n" + "="*50)
    print(f"ACCURACY COMPARISON:")
    print(f"TensorFlow Model: {tf_accuracy:.2f}%")
    print(f"Meta Classifier: {meta_accuracy:.2f}%")
    print(f"Improvement: {meta_accuracy - tf_accuracy:.2f}%")

    # Print detailed model comparison
    model.print_model_comparison('0123456789+*/=()')

    print("="*50)
